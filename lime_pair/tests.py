"""
Unit tests for the lime_pair module.

Tests cover:
- PairIndexedString initialization and dual vocabulary building
- Segment-aware word removal
- PairLimeTextExplainer mode validation and perturbation behavior
"""

import unittest
import numpy as np
from unittest.mock import MagicMock

from .indexed_string import PairIndexedString
from .explainer import PairLimeTextExplainer


class TestPairIndexedString(unittest.TestCase):
    """Tests for PairIndexedString class."""

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
        idx = PairIndexedString(self.simple_text)

        self.assertEqual(idx.num_words_left(), 2)  # Hello, world
        self.assertEqual(idx.num_words_right(), 2)  # Goodbye, world
        self.assertEqual(idx.num_words(), 4)

    def test_init_missing_separator_raises(self):
        """Test that missing separator raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            PairIndexedString('No separator here')

        self.assertIn('Separator', str(ctx.exception))

    def test_init_custom_separator(self):
        """Test initialization with custom separator.

        Note: The separator should be all non-word characters to ensure it
        stays as one token when the default regex tokenizer splits on \\W+.
        """
        text = 'Left text <###> Right text'
        idx = PairIndexedString(text, separator='<###>')

        self.assertEqual(idx.num_words_left(), 2)
        self.assertEqual(idx.num_words_right(), 2)

    def test_dual_vocabularies_bow(self):
        """Test that BOW mode creates separate vocabularies."""
        idx = PairIndexedString(self.simple_text, bow=True)

        # 'world' should appear in both vocabularies as separate features
        self.assertIn('world', idx.vocab_left)
        self.assertIn('world', idx.vocab_right)

        # They should have different global positions
        self.assertEqual(idx.vocab_left['world'], 1)  # Second word in left
        self.assertEqual(idx.vocab_right['world'], 1)  # Second word in right

    def test_separator_index(self):
        """Test separator index is correctly identified."""
        idx = PairIndexedString(self.simple_text, bow=True)

        # Separator index should be the size of left vocabulary
        self.assertEqual(idx.get_separator_index(), 2)

    def test_has_right_segment(self):
        """Test detection of right segment."""
        idx = PairIndexedString(self.simple_text)
        self.assertTrue(idx.has_right_segment())

    def test_inverse_removing_left(self):
        """Test removing words from left segment only."""
        idx = PairIndexedString(self.simple_text, bow=True)

        # Remove 'Hello' (index 0 in left vocab)
        result = idx.inverse_removing([0], segment='left')

        # 'Hello' should be gone, but right segment intact
        self.assertNotIn('Hello', result.split('$&*&*&$')[0])
        self.assertIn('Goodbye', result)
        self.assertIn('world', result.split('$&*&*&$')[1])

    def test_inverse_removing_right(self):
        """Test removing words from right segment only."""
        idx = PairIndexedString(self.simple_text, bow=True)

        # Remove 'Goodbye' (index 0 in right vocab)
        result = idx.inverse_removing([0], segment='right')

        # 'Goodbye' should be gone, but left segment intact
        self.assertIn('Hello', result)
        self.assertNotIn('Goodbye', result)

    def test_inverse_removing_invalid_segment_raises(self):
        """Test that invalid segment raises ValueError."""
        idx = PairIndexedString(self.simple_text)

        with self.assertRaises(ValueError) as ctx:
            idx.inverse_removing([0], segment='invalid')

        self.assertIn('left', str(ctx.exception))
        self.assertIn('right', str(ctx.exception))

    def test_non_bow_mode(self):
        """Test non-BOW (position-aware) mode."""
        text = 'word word $&*&*&$ word word'
        idx = PairIndexedString(text, bow=False)

        # Each 'word' should be a separate feature
        self.assertEqual(idx.num_words_left(), 2)  # Two positions in left
        self.assertEqual(idx.num_words_right(), 2)  # Two positions in right

    def test_complex_text_vocabularies(self):
        """Test vocabulary building with complex text."""
        idx = PairIndexedString(self.complex_text, bow=True)

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
        idx = PairIndexedString(self.simple_text, bow=True)

        self.assertEqual(idx.word(0, segment='left'), 'Hello')
        self.assertEqual(idx.word(0, segment='right'), 'Goodbye')

    def test_repr(self):
        """Test string representation."""
        idx = PairIndexedString(self.simple_text)
        repr_str = repr(idx)

        self.assertIn('PairIndexedString', repr_str)
        self.assertIn('left_vocab=2', repr_str)
        self.assertIn('right_vocab=2', repr_str)


class TestPairLimeTextExplainer(unittest.TestCase):
    """Tests for PairLimeTextExplainer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.text_pair = 'Hello world $&*&*&$ Goodbye world'

        # Mock classifier that returns deterministic probabilities based on text
        # This avoids NaN issues from purely random values
        def mock_classifier(texts):
            probs = []
            for text in texts:
                # Base probability on text length - more words = higher class 1
                word_count = len(text.split())
                # Ensure probabilities are bounded away from 0 and 1
                p1 = 0.1 + 0.8 * min(word_count / 10.0, 1.0)
                probs.append([1 - p1, p1])
            return np.array(probs)

        self.mock_classifier = mock_classifier

    def test_init_valid_modes(self):
        """Test initialization with valid modes."""
        for mode in ['left', 'right']:
            exp = PairLimeTextExplainer(mode=mode, bow=True)
            self.assertEqual(exp.get_mode(), mode)

        # rand mode only valid for non-BOW
        exp = PairLimeTextExplainer(mode='rand', bow=False)
        self.assertEqual(exp.get_mode(), 'rand')

    def test_init_invalid_mode_raises(self):
        """Test that invalid mode raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            PairLimeTextExplainer(mode='invalid')

        self.assertIn('mode', str(ctx.exception))

    def test_rand_mode_bow_raises(self):
        """Test that rand mode with BOW raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            PairLimeTextExplainer(mode='rand', bow=True)

        self.assertIn('rand', str(ctx.exception))
        self.assertIn('bow', str(ctx.exception).lower())

    def test_custom_separator(self):
        """Test custom separator configuration."""
        exp = PairLimeTextExplainer(separator='[SEP]')
        self.assertEqual(exp.get_separator(), '[SEP]')

    def test_explain_instance_returns_explanation(self):
        """Test that explain_instance returns an explanation object."""
        exp = PairLimeTextExplainer(mode='left', bow=True, random_state=42)

        explanation = exp.explain_instance(
            self.text_pair,
            self.mock_classifier,
            num_samples=100,
            num_features=5
        )

        # Check explanation has expected attributes
        self.assertTrue(hasattr(explanation, 'local_exp'))
        self.assertTrue(hasattr(explanation, 'predict_proba'))

    def test_left_mode_only_perturbs_left(self):
        """Test that left mode only perturbs the left segment."""
        exp = PairLimeTextExplainer(mode='left', bow=True, random_state=42)

        # Track which texts are passed to classifier
        texts_seen = []

        def tracking_classifier(texts):
            texts_seen.extend(texts)
            return np.random.rand(len(texts), 2)

        exp.explain_instance(
            self.text_pair,
            tracking_classifier,
            num_samples=50,
            num_features=5
        )

        # Check that right segment is always intact
        for text in texts_seen[1:]:  # Skip original
            right_part = text.split('$&*&*&$')[1]
            self.assertIn('Goodbye', right_part)
            self.assertIn('world', right_part)

    def test_right_mode_only_perturbs_right(self):
        """Test that right mode only perturbs the right segment."""
        exp = PairLimeTextExplainer(mode='right', bow=True, random_state=42)

        texts_seen = []

        def tracking_classifier(texts):
            texts_seen.extend(texts)
            return np.random.rand(len(texts), 2)

        exp.explain_instance(
            self.text_pair,
            tracking_classifier,
            num_samples=50,
            num_features=5
        )

        # Check that left segment is always intact
        for text in texts_seen[1:]:  # Skip original
            left_part = text.split('$&*&*&$')[0]
            self.assertIn('Hello', left_part)
            self.assertIn('world', left_part)

    def test_non_bow_mode_left(self):
        """Test non-BOW mode with left perturbation."""
        exp = PairLimeTextExplainer(mode='left', bow=False, random_state=42)

        explanation = exp.explain_instance(
            self.text_pair,
            self.mock_classifier,
            num_samples=50,
            num_features=5
        )

        self.assertTrue(hasattr(explanation, 'local_exp'))

    def test_non_bow_mode_rand(self):
        """Test non-BOW mode with random perturbation."""
        exp = PairLimeTextExplainer(mode='rand', bow=False, random_state=42)

        explanation = exp.explain_instance(
            self.text_pair,
            self.mock_classifier,
            num_samples=50,
            num_features=5
        )

        self.assertTrue(hasattr(explanation, 'local_exp'))

    def test_reproducibility_with_random_state(self):
        """Test that random_state ensures reproducibility."""
        exp1 = PairLimeTextExplainer(mode='left', bow=True, random_state=42)
        exp2 = PairLimeTextExplainer(mode='left', bow=True, random_state=42)

        # Use deterministic classifier that varies with input
        def deterministic_classifier(texts):
            probs = []
            for text in texts:
                # Vary probability based on text content
                word_count = len(text.split())
                p1 = 0.2 + 0.6 * min(word_count / 8.0, 1.0)
                probs.append([1 - p1, p1])
            return np.array(probs)

        explanation1 = exp1.explain_instance(
            self.text_pair,
            deterministic_classifier,
            num_samples=50,
            num_features=5
        )

        explanation2 = exp2.explain_instance(
            self.text_pair,
            deterministic_classifier,
            num_samples=50,
            num_features=5
        )

        # Local explanations should be identical
        for label in explanation1.local_exp:
            exp1_features = dict(explanation1.local_exp[label])
            exp2_features = dict(explanation2.local_exp[label])
            self.assertEqual(exp1_features, exp2_features)

    def test_repr(self):
        """Test string representation."""
        exp = PairLimeTextExplainer(mode='left', bow=True)
        repr_str = repr(exp)

        self.assertIn('PairLimeTextExplainer', repr_str)
        self.assertIn('left', repr_str)
        self.assertIn('bow=True', repr_str)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""

    def test_full_pipeline_bow(self):
        """Test complete pipeline in BOW mode."""
        text = (
            'The quick brown fox jumps over the lazy dog. '
            '$&*&*&$'
            'A quick brown dog runs under the lazy fox.'
        )

        def mock_classifier(texts):
            # Simple classifier: more words = higher class 1 probability
            # Ensure probabilities are bounded away from 0 and 1
            probs = []
            for t in texts:
                word_count = len(t.split())
                p1 = 0.1 + 0.8 * min(word_count / 20.0, 1.0)
                probs.append([1 - p1, p1])
            return np.array(probs)

        # Test both modes
        for mode in ['left', 'right']:
            exp = PairLimeTextExplainer(
                mode=mode, bow=True, random_state=42
            )

            explanation = exp.explain_instance(
                text,
                mock_classifier,
                num_samples=100,
                num_features=5,
                labels=(0, 1)
            )

            # Should have explanations for both labels
            self.assertIn(0, explanation.local_exp)
            self.assertIn(1, explanation.local_exp)

            # Should have non-zero weights
            weights_0 = [w for _, w in explanation.local_exp[0]]
            weights_1 = [w for _, w in explanation.local_exp[1]]

            self.assertTrue(any(w != 0 for w in weights_0))
            self.assertTrue(any(w != 0 for w in weights_1))

    def test_full_pipeline_non_bow(self):
        """Test complete pipeline in non-BOW mode."""
        text = 'Word word $&*&*&$ word word'

        def mock_classifier(texts):
            # Deterministic classifier based on text length
            probs = []
            for t in texts:
                word_count = len(t.split())
                p1 = 0.2 + 0.6 * min(word_count / 6.0, 1.0)
                probs.append([1 - p1, p1])
            return np.array(probs)

        for mode in ['left', 'right', 'rand']:
            exp = PairLimeTextExplainer(
                mode=mode, bow=False, random_state=42
            )

            explanation = exp.explain_instance(
                text,
                mock_classifier,
                num_samples=50,
                num_features=3
            )

            self.assertTrue(hasattr(explanation, 'local_exp'))


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestPairIndexedString))
    suite.addTests(loader.loadTestsFromTestCase(TestPairLimeTextExplainer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    unittest.main()
