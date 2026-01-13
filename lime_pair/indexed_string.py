"""
PairIndexedString: String indexing for pairwise text classification.

This module provides a customized IndexedString class that handles text pairs
separated by a special token, building independent vocabularies for each segment.
"""

import itertools
import logging
import re
from typing import Callable, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class PairIndexedString:
    """
    String indexer for pairwise text classification with dual vocabularies.

    This class extends the concept of LIME's IndexedString to handle text pairs
    by building separate vocabularies for each segment (left and right of separator).

    The key insight is that for pairwise authorship verification:
    - The same word appearing in both texts should be treated as TWO different features
    - This allows independent perturbation of each text segment
    - Enables controlling similarity (not just dissimilarity) through perturbations

    Parameters
    ----------
    raw_string : str
        The combined text pair with separator token.
    separator : str, default="$&*&*&$"
        Token that separates the two text segments. Must be unique and not
        appear naturally in the text.
    split_expression : str or callable, default=r'\\W+'
        Regex pattern or callable for tokenization. If regex, used with re.split.
        If callable, should return a list of tokens.
    bow : bool, default=True
        If True (bag of words), a word is the same everywhere within its segment.
        If False, each word position is treated as a separate feature.
    mask_string : str, optional
        String to replace removed words with when bow=False.
        Defaults to 'UNKWORDZ'.

    Attributes
    ----------
    vocab_left : dict
        Vocabulary mapping for the left text segment {word: index}.
    vocab_right : dict
        Vocabulary mapping for the right text segment {word: index}.
    inverse_vocab_left : list
        List of words in left vocabulary, indexed by their vocabulary index.
    inverse_vocab_right : list
        List of words in right vocabulary, indexed by their vocabulary index.
    positions_left : list or np.ndarray
        Token positions for left vocabulary words.
    positions_right : list or np.ndarray
        Token positions for right vocabulary words.
    separator_index : int
        The vocabulary index where the separator was found (marks boundary).

    Raises
    ------
    ValueError
        If the separator is not found in the input string.

    Example
    -------
    >>> text = 'Hello world $&*&*&$ Goodbye world'
    >>> idx_str = PairIndexedString(text)
    >>> idx_str.num_words()  # Total features across both segments
    4
    >>> idx_str.inverse_vocab_left
    ['Hello', 'world']
    >>> idx_str.inverse_vocab_right
    ['Goodbye', 'world']
    """

    def __init__(
        self,
        raw_string: str,
        separator: str = "$&*&*&$",
        split_expression: Union[str, Callable] = r'\W+',
        bow: bool = True,
        mask_string: Optional[str] = None
    ):
        self.raw = raw_string
        self.separator = separator
        self.mask_string = mask_string if mask_string is not None else 'UNKWORDZ'
        self.bow = bow

        # Validate separator exists
        if separator not in raw_string:
            raise ValueError(
                f"Separator '{separator}' not found in input string. "
                "Ensure the text pair is properly formatted with the separator token."
            )

        # Tokenize the string
        if callable(split_expression):
            tokens = split_expression(self.raw)
            self.as_list = self._segment_with_tokens(self.raw, tokens)
            tokens_set = set(tokens)
            non_word = lambda s: s not in tokens_set
        else:
            splitter = re.compile(r'(%s)|$' % split_expression)
            self.as_list = [s for s in splitter.split(self.raw) if s]
            non_word = splitter.match

        self.as_np = np.array(self.as_list)
        self.string_start = np.hstack(
            ([0], np.cumsum([len(x) for x in self.as_np[:-1]]))
        )

        # Initialize dual vocabularies
        self.vocab_left = {}
        self.vocab_right = {}
        self.inverse_vocab_left = []
        self.inverse_vocab_right = []
        self.positions_left = []
        self.positions_right = []

        # Track separator position
        self.separator_index = None
        self._separator_found = False

        # Build vocabularies
        non_vocab = set()
        for i, word in enumerate(self.as_np):
            # Check for separator
            if self.separator in word:
                self.separator_index = len(self.inverse_vocab_left)
                self._separator_found = True
                logger.debug(f"Separator found at token position {i}, vocab index {self.separator_index}")
                continue

            # Skip non-words and already-seen non-words
            if word in non_vocab:
                continue
            if non_word(word):
                non_vocab.add(word)
                continue

            # Add to appropriate vocabulary
            if bow:
                self._add_word_bow(word, i)
            else:
                self._add_word_position(word, i)

        # Convert positions to numpy array for non-BOW mode
        if not bow:
            self.positions_left = np.array(self.positions_left)
            self.positions_right = np.array(self.positions_right)

        # Set separator index if not found (all text is "left")
        if self.separator_index is None:
            self.separator_index = len(self.inverse_vocab_left)
            logger.warning(
                "Separator token detected in string but not as separate token. "
                "This may indicate tokenization issues."
            )

        logger.debug(
            f"Indexed string: {self.num_words_left()} left features, "
            f"{self.num_words_right()} right features"
        )

    def _add_word_bow(self, word: str, position: int) -> None:
        """Add a word in bag-of-words mode to the appropriate vocabulary."""
        if not self._separator_found:
            # Left segment
            if word not in self.vocab_left:
                self.vocab_left[word] = len(self.vocab_left)
                self.inverse_vocab_left.append(word)
                self.positions_left.append([])
            idx = self.vocab_left[word]
            self.positions_left[idx].append(position)
        else:
            # Right segment
            if word not in self.vocab_right:
                self.vocab_right[word] = len(self.vocab_right)
                self.inverse_vocab_right.append(word)
                self.positions_right.append([])
            idx = self.vocab_right[word]
            self.positions_right[idx].append(position)

    def _add_word_position(self, word: str, position: int) -> None:
        """Add a word in position-aware mode to the appropriate vocabulary."""
        if not self._separator_found:
            self.inverse_vocab_left.append(word)
            self.positions_left.append(position)
        else:
            self.inverse_vocab_right.append(word)
            self.positions_right.append(position)

    @staticmethod
    def _segment_with_tokens(text: str, tokens: List[str]) -> List[str]:
        """Segment a string around tokens from a custom tokenizer."""
        list_form = []
        text_ptr = 0
        for token in tokens:
            inter_token_string = []
            while not text[text_ptr:].startswith(token):
                inter_token_string.append(text[text_ptr])
                text_ptr += 1
                if text_ptr >= len(text):
                    raise ValueError(
                        "Tokenization produced tokens that do not belong in string!"
                    )
            text_ptr += len(token)
            if inter_token_string:
                list_form.append(''.join(inter_token_string))
            list_form.append(token)
        if text_ptr < len(text):
            list_form.append(text[text_ptr:])
        return list_form

    def raw_string(self) -> str:
        """Return the original raw string."""
        return self.raw

    def num_words(self) -> int:
        """Return total number of features across both segments."""
        return self.num_words_left() + self.num_words_right()

    def num_words_left(self) -> int:
        """Return number of features in left segment."""
        return len(self.inverse_vocab_left)

    def num_words_right(self) -> int:
        """Return number of features in right segment."""
        return len(self.inverse_vocab_right)

    def get_separator_index(self) -> int:
        """Return the vocabulary index marking the boundary between segments."""
        return self.separator_index

    def has_right_segment(self) -> bool:
        """Check if the string has a non-empty right segment."""
        return len(self.inverse_vocab_right) > 0

    def word(self, idx: int, segment: Optional[str] = None) -> str:
        """
        Return the word at the given vocabulary index.

        Parameters
        ----------
        idx : int
            Vocabulary index. If segment is None, this is treated as a global
            index where 0 to sep-1 are left segment, sep to num_words-1 are right.
        segment : str, optional
            Which segment's vocabulary to use ('left' or 'right').
            If None, automatically determines based on idx.

        Returns
        -------
        str
            The word at the given index.
        """
        if segment is None:
            # Auto-detect segment based on global index
            if idx < self.separator_index:
                return self.inverse_vocab_left[idx]
            else:
                right_idx = idx - self.separator_index
                return self.inverse_vocab_right[right_idx]
        elif segment == 'left':
            return self.inverse_vocab_left[idx]
        elif segment == 'right':
            return self.inverse_vocab_right[idx]
        else:
            raise ValueError(f"segment must be 'left' or 'right', got '{segment}'")

    def string_position(self, idx: int, segment: str = 'left') -> np.ndarray:
        """
        Return character positions for a vocabulary index.

        Parameters
        ----------
        idx : int
            Vocabulary index.
        segment : str, default='left'
            Which segment's vocabulary to use.

        Returns
        -------
        np.ndarray
            Array of character start positions.
        """
        if segment == 'left':
            positions = self.positions_left
        elif segment == 'right':
            positions = self.positions_right
        else:
            raise ValueError(f"segment must be 'left' or 'right', got '{segment}'")

        if self.bow:
            return self.string_start[positions[idx]]
        else:
            return self.string_start[[positions[idx]]]

    def inverse_removing(
        self,
        words_to_remove: List[int],
        segment: str = 'left'
    ) -> str:
        """
        Return string with specified words removed from one segment.

        This is the key method for pair-aware perturbations. It removes words
        only from the specified segment, leaving the other segment intact.

        Parameters
        ----------
        words_to_remove : list of int
            Vocabulary indices to remove (within the specified segment).
        segment : str, default='left'
            Which segment to remove words from ('left' or 'right').

        Returns
        -------
        str
            The string with specified words removed/masked.

        Raises
        ------
        ValueError
            If segment is not 'left' or 'right'.
        """
        if segment not in ('left', 'right'):
            raise ValueError(f"segment must be 'left' or 'right', got '{segment}'")

        mask = np.ones(self.as_np.shape[0], dtype=bool)

        if segment == 'left':
            indices = self._get_token_indices(words_to_remove, self.positions_left)
        else:
            indices = self._get_token_indices(words_to_remove, self.positions_right)

        mask[indices] = False

        if not self.bow:
            return ''.join([
                self.as_list[i] if mask[i] else self.mask_string
                for i in range(mask.shape[0])
            ])
        return ''.join([self.as_list[v] for v in mask.nonzero()[0]])

    def _get_token_indices(
        self,
        vocab_indices: List[int],
        positions: Union[List[List[int]], np.ndarray]
    ) -> List[int]:
        """Convert vocabulary indices to token indices."""
        if self.bow:
            return list(itertools.chain.from_iterable(
                [positions[z] for z in vocab_indices]
            ))
        else:
            return positions[vocab_indices].tolist()

    def __repr__(self) -> str:
        return (
            f"PairIndexedString(left_vocab={self.num_words_left()}, "
            f"right_vocab={self.num_words_right()}, bow={self.bow})"
        )
