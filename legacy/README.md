# Legacy Files

This folder contains original development notebooks that have been superseded by
the `lime_pair` module.

## Files

### lime22notes.ipynb

Original experimental notebook containing the development of pair-aware LIME
text explanations for authorship verification.

This notebook contains:
- Early experiments with dual vocabulary approaches
- Multiple versions of `MyIndexedString` and `MyLimeTextExplainer` classes
- Debug output and intermediate development steps
- Both "Old" and newer implementations mixed together

**Note**: This notebook is kept for reference only. The clean, production-ready
implementation is now in the `lime_pair/` module.

## Migration

The functionality from this notebook has been refactored into:

- `lime_pair/indexed_string.py` - `PairIndexedString` class
- `lime_pair/explainer.py` - `PairLimeTextExplainer` class
- `lime_pair_demo.ipynb` - Clean demonstration notebook

See the main `lime_pair/` module for the current implementation.
