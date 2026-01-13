"""
PairLimeTextExplainer: LIME explainer for pairwise text classification.

This module provides a customized LIME text explainer that handles text pairs
with independent perturbation strategies for each segment.
"""

import logging
from functools import partial
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import scipy as sp
import sklearn.metrics
from sklearn.utils import check_random_state

from lime import lime_base
from lime.lime_text import IndexedCharacters, TextDomainMapper, LimeTextExplainer
from lime import explanation

from .indexed_string import PairIndexedString

logger = logging.getLogger(__name__)


# Type alias for perturbation mode
PerturbationMode = Literal['left', 'right', 'rand']


class PairLimeTextExplainer(LimeTextExplainer):
    """
    LIME Text Explainer adapted for pairwise text classification.

    This explainer extends LimeTextExplainer to handle text pairs by implementing
    pair-aware perturbation strategies. The key innovation is the ability to
    perturb each text segment independently, which enables:

    - Explaining class 1 (same author) by controlling similarity
    - Meaningful feature weights for both classes
    - Independent analysis of each text's contribution

    Perturbation Modes
    ------------------
    - 'left': Only perturb the first text in the pair. Useful for understanding
      how the left text contributes to similarity/dissimilarity.
    - 'right': Only perturb the second text in the pair. Useful for understanding
      how the right text contributes to similarity/dissimilarity.
    - 'rand': For each perturbation sample, randomly choose which side to alter.
      This mode ensures both texts are involved equally on average while each
      individual perturbation affects only one text. Note: only available for
      non-BOW mode due to architectural constraints.

    Theoretical Background
    ----------------------
    Standard LIME perturbations can only control dissimilarity:
    - Removing a word with different frequency in two texts → more similar
    - Removing a word with similar frequency → similarity unchanged

    Pair-aware perturbations can control similarity:
    - Removing a word only from one text → breaks any similarity that existed

    Therefore:
    - Use standard LIME for class 0 (different authors) explanations
    - Use PairLimeTextExplainer for class 1 (same author) explanations

    Parameters
    ----------
    mode : {'left', 'right', 'rand'}, default='left'
        Perturbation mode determining which segment(s) to perturb.
        'rand' is only available when bow=False.
    separator : str, default='$&*&*&$'
        Token separating the two text segments in the input.
    kernel_width : float, default=25
        Width of the exponential kernel for weighting samples.
    kernel : callable, optional
        Similarity kernel function. Defaults to exponential kernel.
    verbose : bool, default=False
        If True, print local prediction values from the linear model.
    class_names : list of str, optional
        Names of the classes for display purposes.
    feature_selection : str, default='highest_weights'
        Method for feature selection in the linear model. Defaults to
        'highest_weights' instead of 'auto' because pair-aware perturbation
        creates constant features (the non-perturbed segment), which causes
        forward selection to fail with numerical errors.
    split_expression : str or callable, default=r'\\W+'
        Regex or callable for tokenization.
    bow : bool, default=True
        If True, use bag-of-words mode (word frequency matters).
        If False, use position-aware mode (word position matters).
    mask_string : str, optional
        String to use for masking words in non-BOW mode.
    random_state : int or RandomState, optional
        Random state for reproducibility.
    char_level : bool, default=False
        If True, treat each character as a feature.

    Raises
    ------
    ValueError
        If mode='rand' is used with bow=True.

    Example
    -------
    >>> from lime_pair import PairLimeTextExplainer
    >>>
    >>> # For class 1 explanations, use both left and right explainers
    >>> explainer_left = PairLimeTextExplainer(bow=True, mode='left')
    >>> explainer_right = PairLimeTextExplainer(bow=True, mode='right')
    >>>
    >>> text_pair = "First text here $&*&*&$ Second text here"
    >>> exp_left = explainer_left.explain_instance(text_pair, classifier_fn)
    >>> exp_right = explainer_right.explain_instance(text_pair, classifier_fn)

    Notes
    -----
    For bag-of-words mode, the 'rand' mode is not implemented because the
    vocabulary must be determined at initialization time, and switching
    between vocabularies during perturbation would break the distance
    metric calculation.
    """

    def __init__(
        self,
        mode: PerturbationMode = 'left',
        separator: str = '$&*&*&$',
        kernel_width: float = 25,
        kernel: Optional[Callable] = None,
        verbose: bool = False,
        class_names: Optional[List[str]] = None,
        feature_selection: str = 'highest_weights',
        split_expression: Union[str, Callable] = r'\W+',
        bow: bool = True,
        mask_string: Optional[str] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        char_level: bool = False
    ):
        # Validate mode
        if mode not in ('left', 'right', 'rand'):
            raise ValueError(
                f"mode must be 'left', 'right', or 'rand', got '{mode}'"
            )
        if mode == 'rand' and bow:
            raise ValueError(
                "mode='rand' is not supported for bag-of-words (bow=True). "
                "Use 'left' or 'right' mode, or set bow=False. "
                "This limitation exists because BOW mode requires vocabulary "
                "to be determined at initialization, and the distance metric "
                "calculation depends on consistent feature indexing."
            )

        self.mode = mode
        self.separator = separator

        # Set up kernel
        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        # Initialize state
        self.random_state = check_random_state(random_state)
        self.base = lime_base.LimeBase(
            kernel_fn, verbose, random_state=self.random_state
        )
        self.class_names = class_names
        self.vocabulary = None
        self.feature_selection = feature_selection
        self.bow = bow
        self.mask_string = mask_string
        self.split_expression = split_expression
        self.char_level = char_level

        logger.debug(
            f"Initialized PairLimeTextExplainer: mode={mode}, bow={bow}, "
            f"separator='{separator}'"
        )

    def explain_instance(
        self,
        text_instance: str,
        classifier_fn: Callable,
        labels: Tuple[int, ...] = (1,),
        top_labels: Optional[int] = None,
        num_features: int = 10,
        num_samples: int = 5000,
        distance_metric: str = 'cosine',
        model_regressor=None
    ):
        """
        Generate explanations for a text pair prediction.

        Parameters
        ----------
        text_instance : str
            The text pair to explain, with segments separated by self.separator.
        classifier_fn : callable
            Prediction function that takes a list of strings and returns
            prediction probabilities as a (n_samples, n_classes) array.
        labels : tuple of int, default=(1,)
            Labels to explain.
        top_labels : int, optional
            If set, explain the top K predicted labels instead.
        num_features : int, default=10
            Maximum number of features in the explanation.
        num_samples : int, default=5000
            Number of perturbed samples to generate.
        distance_metric : str, default='cosine'
            Distance metric for weighting samples.
        model_regressor : sklearn regressor, optional
            Custom regressor for the local model.

        Returns
        -------
        explanation.Explanation
            LIME explanation object with feature weights.

        Raises
        ------
        ValueError
            If the separator is not found in the text instance.
        """
        # Create indexed string
        if self.char_level:
            indexed_string = IndexedCharacters(
                text_instance, bow=self.bow, mask_string=self.mask_string
            )
        else:
            indexed_string = PairIndexedString(
                text_instance,
                separator=self.separator,
                bow=self.bow,
                split_expression=self.split_expression,
                mask_string=self.mask_string
            )

        domain_mapper = TextDomainMapper(indexed_string)

        # Generate perturbed samples and get predictions
        data, yss, distances = self._data_labels_distances(
            indexed_string, classifier_fn, num_samples,
            distance_metric=distance_metric
        )

        # Set up class names
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]

        # Create explanation object
        ret_exp = explanation.Explanation(
            domain_mapper=domain_mapper,
            class_names=self.class_names,
            random_state=self.random_state
        )
        ret_exp.predict_proba = yss[0]

        # Determine labels to explain
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()

        # Fit local linear model for each label
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score,
             ret_exp.local_pred) = self.base.explain_instance_with_data(
                data, yss, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection
            )

        return ret_exp

    def _data_labels_distances(
        self,
        indexed_string: PairIndexedString,
        classifier_fn: Callable,
        num_samples: int,
        distance_metric: str = 'cosine'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate perturbed samples with pair-aware perturbation strategy.

        This method implements the core pair-aware perturbation logic:
        - For 'left' mode: only perturb left segment features
        - For 'right' mode: only perturb right segment features
        - For 'rand' mode: randomly choose segment per sample (non-BOW only)

        Parameters
        ----------
        indexed_string : PairIndexedString
            The indexed representation of the text pair.
        classifier_fn : callable
            Prediction function.
        num_samples : int
            Number of samples to generate.
        distance_metric : str
            Distance metric for sample weighting.

        Returns
        -------
        data : np.ndarray
            Binary matrix (num_samples, num_features) indicating which
            features are active in each sample.
        labels : np.ndarray
            Prediction probabilities for each sample.
        distances : np.ndarray
            Distance of each sample from the original.
        """
        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric
            ).ravel() * 100

        doc_size = indexed_string.num_words()
        sep = indexed_string.get_separator_index()
        has_right = indexed_string.has_right_segment()

        logger.debug(
            f"Generating {num_samples} samples: doc_size={doc_size}, "
            f"sep={sep}, mode={self.mode}"
        )

        # Initialize data matrix
        data = np.ones((num_samples, doc_size))
        inverse_data = [indexed_string.raw_string()]

        if self.bow:
            data, inverse_data = self._perturb_bow(
                indexed_string, data, inverse_data, num_samples, sep, doc_size, has_right
            )
        else:
            data, inverse_data = self._perturb_non_bow(
                indexed_string, data, inverse_data, num_samples, sep, doc_size, has_right
            )

        # Get predictions
        labels = classifier_fn(inverse_data)
        distances = distance_fn(sp.sparse.csr_matrix(data))

        return data, labels, distances

    def _perturb_bow(
        self,
        indexed_string: PairIndexedString,
        data: np.ndarray,
        inverse_data: List[str],
        num_samples: int,
        sep: int,
        doc_size: int,
        has_right: bool
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate perturbations in bag-of-words mode.

        In BOW mode, we have separate vocabularies for left and right segments.
        Perturbations are applied only to the specified segment.
        """
        if not has_right:
            # No right segment - use standard perturbation
            logger.debug("No right segment found, using standard perturbation")
            return self._perturb_standard(
                indexed_string, data, inverse_data, num_samples, doc_size
            )

        # Calculate sizes for each segment
        left_size = sep
        right_size = doc_size - sep

        logger.debug(f"BOW mode: left_size={left_size}, right_size={right_size}")

        if self.mode == 'left':
            sample_sizes = self.random_state.randint(1, left_size + 1, num_samples - 1)
            features_range = range(0, left_size)
            segment = 'left'
        elif self.mode == 'right':
            sample_sizes = self.random_state.randint(1, right_size + 1, num_samples - 1)
            features_range = range(0, right_size)
            segment = 'right'
        else:
            # 'rand' mode should have been caught in __init__ for BOW
            raise RuntimeError("rand mode should not be reachable for BOW")

        for i, size in enumerate(sample_sizes, start=1):
            inactive = self.random_state.choice(
                list(features_range), size, replace=False
            )

            if self.mode == 'left':
                # Mark left features as inactive in data matrix
                data[i, inactive] = 0
            else:  # right
                # Mark right features as inactive (offset by sep in data matrix)
                data[i, sep + inactive] = 0

            inverse_data.append(
                indexed_string.inverse_removing(inactive.tolist(), segment=segment)
            )

        return data, inverse_data

    def _perturb_non_bow(
        self,
        indexed_string: PairIndexedString,
        data: np.ndarray,
        inverse_data: List[str],
        num_samples: int,
        sep: int,
        doc_size: int,
        has_right: bool
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate perturbations in non-bag-of-words (position-aware) mode.

        In non-BOW mode, each word position is a separate feature.
        The 'rand' mode is available here.
        """
        if not has_right or sep >= doc_size:
            logger.debug("No right segment or empty, using standard perturbation")
            return self._perturb_standard(
                indexed_string, data, inverse_data, num_samples, doc_size
            )

        features_range_left = range(0, sep)
        features_range_right = range(sep, doc_size)

        sample_left = self.random_state.randint(1, sep + 1, num_samples - 1)
        sample_right = self.random_state.randint(1, doc_size - sep + 1, num_samples - 1)

        if self.mode == 'left':
            logger.debug("Non-BOW left mode")
            for i, size in enumerate(sample_left, start=1):
                inactive = self.random_state.choice(
                    list(features_range_left), size, replace=False
                )
                data[i, inactive] = 0
                inverse_data.append(
                    indexed_string.inverse_removing(inactive.tolist(), segment='left')
                )

        elif self.mode == 'right':
            logger.debug("Non-BOW right mode")
            for i, size in enumerate(sample_right, start=1):
                inactive = self.random_state.choice(
                    list(features_range_right), size, replace=False
                )
                data[i, inactive] = 0
                # For non-BOW, we use global indices but need to map to right segment
                # The indices are already global (sep to doc_size)
                local_inactive = (inactive - sep).tolist()
                inverse_data.append(
                    indexed_string.inverse_removing(local_inactive, segment='right')
                )

        else:  # rand mode
            logger.debug("Non-BOW rand mode")
            # Randomly decide which side to perturb for each sample
            side_choices = self.random_state.randint(2, size=num_samples - 1)

            for i in range(num_samples - 1):
                if side_choices[i]:  # perturb right
                    inactive = self.random_state.choice(
                        list(features_range_right), sample_right[i], replace=False
                    )
                    data[i + 1, inactive] = 0
                    local_inactive = (inactive - sep).tolist()
                    inverse_data.append(
                        indexed_string.inverse_removing(local_inactive, segment='right')
                    )
                else:  # perturb left
                    inactive = self.random_state.choice(
                        list(features_range_left), sample_left[i], replace=False
                    )
                    data[i + 1, inactive] = 0
                    inverse_data.append(
                        indexed_string.inverse_removing(inactive.tolist(), segment='left')
                    )

        return data, inverse_data

    def _perturb_standard(
        self,
        indexed_string: PairIndexedString,
        data: np.ndarray,
        inverse_data: List[str],
        num_samples: int,
        doc_size: int
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Standard perturbation when there's no pair structure.

        Falls back to standard LIME behavior when separator is not found
        or right segment is empty.
        """
        features_range = range(doc_size)
        sample_sizes = self.random_state.randint(1, doc_size + 1, num_samples - 1)

        for i, size in enumerate(sample_sizes, start=1):
            inactive = self.random_state.choice(
                list(features_range), size, replace=False
            )
            data[i, inactive] = 0
            inverse_data.append(
                indexed_string.inverse_removing(inactive.tolist(), segment='left')
            )

        return data, inverse_data

    def get_mode(self) -> str:
        """Return the current perturbation mode."""
        return self.mode

    def get_separator(self) -> str:
        """Return the separator token."""
        return self.separator

    def __repr__(self) -> str:
        return (
            f"PairLimeTextExplainer(mode='{self.mode}', bow={self.bow}, "
            f"separator='{self.separator}')"
        )
