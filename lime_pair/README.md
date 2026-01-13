# lime_pair: Pair-aware LIME for Pairwise Text Classification

A customized LIME (Local Interpretable Model-agnostic Explanations) implementation
designed for explaining pairwise text classification models, particularly for
authorship verification tasks.

## Motivation

Standard LIME perturbations can only control **dissimilarity** between texts:
- Removing a word with different frequency in two texts → texts become more similar
- Removing a word with similar frequency → similarity unchanged

This means standard LIME works well for explaining class 0 (different authors) but
produces meaningless explanations for class 1 (same author).

**Pair-aware LIME** solves this by perturbing each text segment independently,
allowing control over **similarity** as well.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or manually:
pip install numpy scipy scikit-learn lime
```

## Quick Start

```python
from lime_pair import PairLimeTextExplainer

# Create explainers for left and right perturbation modes
explainer_left = PairLimeTextExplainer(mode='left', bow=True)
explainer_right = PairLimeTextExplainer(mode='right', bow=True)

# Your text pair with separator
text_pair = "First text here $&*&*&$ Second text here"

# Generate explanations
exp_left = explainer_left.explain_instance(text_pair, classifier_fn)
exp_right = explainer_right.explain_instance(text_pair, classifier_fn)
```

## Key Components

### PairIndexedString

String indexer that builds **separate vocabularies** for each text segment.

```python
from lime_pair import PairIndexedString

text = "Hello world $&*&*&$ Goodbye world"
idx = PairIndexedString(text)

# Separate vocabularies
print(idx.vocab_left)   # {'Hello': 0, 'world': 1}
print(idx.vocab_right)  # {'Goodbye': 0, 'world': 1}

# Remove words from specific segments
idx.inverse_removing([0], segment='left')   # Removes 'Hello' from left only
idx.inverse_removing([0], segment='right')  # Removes 'Goodbye' from right only
```

### PairLimeTextExplainer

LIME explainer with pair-aware perturbation modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| `'left'` | Only perturb left text | Analyze left text's contribution |
| `'right'` | Only perturb right text | Analyze right text's contribution |
| `'rand'` | Random side per sample | General exploration (non-BOW only) |

## Recommended Usage Pattern

```python
from lime.lime_text import LimeTextExplainer
from lime_pair import PairLimeTextExplainer

# For class 0 (different authors) - use standard LIME
standard_explainer = LimeTextExplainer(bow=True)

# For class 1 (same author) - use pair-aware LIME
explainer_left = PairLimeTextExplainer(mode='left', bow=True)
explainer_right = PairLimeTextExplainer(mode='right', bow=True)

# Choose based on predicted class
if predicted_class == 0:
    explanation = standard_explainer.explain_instance(text_pair, classifier)
else:
    exp_left = explainer_left.explain_instance(text_pair, classifier)
    exp_right = explainer_right.explain_instance(text_pair, classifier)
```

## Configuration

### Custom Separator

```python
PairLimeTextExplainer(separator='[SEP]')  # Use [SEP] instead of $&*&*&$
```

### Logging

```python
import logging
logging.getLogger('lime_pair').setLevel(logging.DEBUG)
```

## Testing

```bash
# Run standalone tests (no LIME dependency required)
python lime_pair/test_indexed_string.py

# Run full test suite (requires LIME)
python -m unittest lime_pair.tests
```

## Module Structure

```
lime_pair/
├── __init__.py           # Package exports
├── indexed_string.py     # PairIndexedString class
├── explainer.py          # PairLimeTextExplainer class
├── tests.py              # Full test suite (requires LIME)
├── test_indexed_string.py # Standalone tests
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Theoretical Background

See the research paper for detailed theoretical analysis of why standard LIME
fails for pairwise classification and how pair-aware perturbations solve this.

Key findings:
- Dissimilarity is controllable through standard perturbations
- Similarity requires segment-independent perturbations
- BOW mode with dual vocabularies provides best results for class 1 explanations
- Intercept values significantly lower (< 0.3 vs > 0.99) with pair-aware approach

## License

MIT License
