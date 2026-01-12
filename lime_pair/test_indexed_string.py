"""
Standalone unit tests for PairIndexedString (no LIME dependency required).

This test file tests the core functionality of the PairIndexedString class
which doesn't depend on the LIME library.
"""

import sys
import unittest

# Add parent directory to path for imports
sys.path.insert(0, '/home/user/BertAA')

import numpy as np


class TestPairIndexedStringStandalone(unittest.TestCase):
    """Tests for PairIndexedString class without LIME dependency."""

    @classmethod
    def setUpClass(cls):
        """Import the module after numpy is available."""
        from lime_pair.indexed_string import PairIndexedString
        cls.PairIndexedString = PairIndexedString

    def setUp(self):
        """Set up test fixtures."""
        self.simple_text = 'Hello world $&*&*&$ Goodbye world'
        self.complex_text = (
            'Rinoa let let out a soft giggle. "Okay Uncle Rinoa Laguna." '
            '$&*&*&$'
            '"As always, make a giggle Rinoa yourselves at a good home!"'
        )

    def test_init_simple_text(self):
        """Test basic initialization with simple text pair."""
        idx = self.PairIndexedString(self.simple_text)

        self.assertEqual(idx.num_words_left(), 2)  # Hello, world
        self.assertEqual(idx.num_words_right(), 2)  # Goodbye, world
        self.assertEqual(idx.num_words(), 4)

    def test_init_missing_separator_raises(self):
        """Test that missing separator raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.PairIndexedString('No separator here')

        self.assertIn('Separator', str(ctx.exception))

    def test_init_custom_separator(self):
        """Test initialization with custom separator.

        Note: The separator should be all non-word characters to ensure it
        stays as one token when the default regex tokenizer splits on \\W+.
        """
        text = 'Left text <###> Right text'
        idx = self.PairIndexedString(text, separator='<###>')

        self.assertEqual(idx.num_words_left(), 2)
        self.assertEqual(idx.num_words_right(), 2)

    def test_dual_vocabularies_bow(self):
        """Test that BOW mode creates separate vocabularies."""
        idx = self.PairIndexedString(self.simple_text, bow=True)

        # 'world' should appear in both vocabularies as separate features
        self.assertIn('world', idx.vocab_left)
        self.assertIn('world', idx.vocab_right)

        # They should have different indices within their vocabularies
        self.assertEqual(idx.vocab_left['world'], 1)  # Second word in left
        self.assertEqual(idx.vocab_right['world'], 1)  # Second word in right

    def test_separator_index(self):
        """Test separator index is correctly identified."""
        idx = self.PairIndexedString(self.simple_text, bow=True)

        # Separator index should be the size of left vocabulary
        self.assertEqual(idx.get_separator_index(), 2)

    def test_has_right_segment(self):
        """Test detection of right segment."""
        idx = self.PairIndexedString(self.simple_text)
        self.assertTrue(idx.has_right_segment())

    def test_inverse_removing_left(self):
        """Test removing words from left segment only."""
        idx = self.PairIndexedString(self.simple_text, bow=True)

        # Remove 'Hello' (index 0 in left vocab)
        result = idx.inverse_removing([0], segment='left')

        # 'Hello' should be gone, but right segment intact
        self.assertNotIn('Hello', result.split('$&*&*&$')[0])
        self.assertIn('Goodbye', result)
        self.assertIn('world', result.split('$&*&*&$')[1])

    def test_inverse_removing_right(self):
        """Test removing words from right segment only."""
        idx = self.PairIndexedString(self.simple_text, bow=True)

        # Remove 'Goodbye' (index 0 in right vocab)
        result = idx.inverse_removing([0], segment='right')

        # 'Goodbye' should be gone, but left segment intact
        self.assertIn('Hello', result)
        self.assertNotIn('Goodbye', result)

    def test_inverse_removing_invalid_segment_raises(self):
        """Test that invalid segment raises ValueError."""
        idx = self.PairIndexedString(self.simple_text)

        with self.assertRaises(ValueError) as ctx:
            idx.inverse_removing([0], segment='invalid')

        self.assertIn('left', str(ctx.exception))
        self.assertIn('right', str(ctx.exception))

    def test_non_bow_mode(self):
        """Test non-BOW (position-aware) mode."""
        text = 'word word $&*&*&$ word word'
        idx = self.PairIndexedString(text, bow=False)

        # Each 'word' should be a separate feature
        self.assertEqual(idx.num_words_left(), 2)  # Two positions in left
        self.assertEqual(idx.num_words_right(), 2)  # Two positions in right

    def test_complex_text_vocabularies(self):
        """Test vocabulary building with complex text."""
        idx = self.PairIndexedString(self.complex_text, bow=True)

        # Check some expected words are in correct vocabularies
        self.assertIn('Rinoa', idx.vocab_left)
        self.assertIn('Rinoa', idx.vocab_right)  # Appears in both
        self.assertIn('giggle', idx.vocab_left)
        self.assertIn('giggle', idx.vocab_right)

        # Words unique to each segment
        self.assertIn('Laguna', idx.vocab_left)
        self.assertNotIn('Laguna', idx.vocab_right)
        self.assertIn('home', idx.vocab_right)
        self.assertNotIn('home', idx.vocab_left)

    def test_word_method(self):
        """Test word retrieval by index."""
        idx = self.PairIndexedString(self.simple_text, bow=True)

        self.assertEqual(idx.word(0, segment='left'), 'Hello')
        self.assertEqual(idx.word(0, segment='right'), 'Goodbye')

    def test_repr(self):
        """Test string representation."""
        idx = self.PairIndexedString(self.simple_text)
        repr_str = repr(idx)

        self.assertIn('PairIndexedString', repr_str)
        self.assertIn('left_vocab=2', repr_str)
        self.assertIn('right_vocab=2', repr_str)

    def test_removing_multiple_words_left(self):
        """Test removing multiple words from left segment."""
        idx = self.PairIndexedString(self.simple_text, bow=True)

        # Remove both 'Hello' (0) and 'world' (1)
        result = idx.inverse_removing([0, 1], segment='left')

        # Left segment should be mostly empty
        left_part = result.split('$&*&*&$')[0]
        self.assertNotIn('Hello', left_part)
        self.assertNotIn('world', left_part)

        # Right segment should be intact
        right_part = result.split('$&*&*&$')[1]
        self.assertIn('Goodbye', right_part)
        self.assertIn('world', right_part)

    def test_raw_string_returns_original(self):
        """Test that raw_string returns the original text."""
        idx = self.PairIndexedString(self.simple_text)
        self.assertEqual(idx.raw_string(), self.simple_text)

    def test_mask_string_non_bow(self):
        """Test custom mask string in non-BOW mode."""
        text = 'word1 word2 $&*&*&$ word3 word4'
        idx = self.PairIndexedString(text, bow=False, mask_string='[MASK]')

        result = idx.inverse_removing([0], segment='left')
        self.assertIn('[MASK]', result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
