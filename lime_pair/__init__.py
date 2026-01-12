"""
Pair-aware LIME Text Explainer for Pairwise Authorship Verification.

This module provides a customized version of LIME (Local Interpretable Model-agnostic
Explanations) designed for explaining pairwise text classification models, particularly
for authorship verification tasks where the input consists of two text excerpts.

Motivation
----------
Standard LIME perturbations can only control dissimilarity between texts (by removing
words with different frequencies). To explain class 1 predictions (same author), we need
to control similarity by perturbing each text independently.

Key insight: The degree of dissimilarity (in dissimilar texts) is controllable through
classical LIME perturbations, but the degree of similarity (in similar texts) is not.

Solution: Pair-aware perturbations that can operate on each text segment independently:
- "left" mode: Only perturb the first text in the pair
- "right" mode: Only perturb the second text in the pair
- "rand" mode: For each perturbation, randomly choose which side to alter (non-BOW only)

Usage
-----
For class 0 (different authors): Use standard LIME
For class 1 (same author): Use PairLimeTextExplainer with 'left' and 'right' modes

Example:
    >>> from lime_pair import PairLimeTextExplainer
    >>> explainer_left = PairLimeTextExplainer(bow=True, mode='left')
    >>> explainer_right = PairLimeTextExplainer(bow=True, mode='right')
    >>> exp_left = explainer_left.explain_instance(text_pair, classifier_fn)
    >>> exp_right = explainer_right.explain_instance(text_pair, classifier_fn)

References
----------
Based on LIME: https://github.com/marcotcr/lime
Adapted for pairwise authorship verification as described in the research.
"""

from .indexed_string import PairIndexedString

# PairLimeTextExplainer requires the lime package
# Import it only if lime is available
try:
    from .explainer import PairLimeTextExplainer
    __all__ = ["PairIndexedString", "PairLimeTextExplainer"]
except ImportError:
    import warnings
    warnings.warn(
        "lime package not found. PairLimeTextExplainer will not be available. "
        "Install lime with: pip install lime"
    )
    PairLimeTextExplainer = None
    __all__ = ["PairIndexedString"]

__version__ = "1.0.0"
