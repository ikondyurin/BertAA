# Patch hardcoded absolute Windows paths in BertAA notebooks and .py files.
# Replaces C:/Users/ivank/Documents/BERT_projects/... with ../BertAA_content/...
# so that BertAA_content only needs to be a sibling of the BertAA folder.
import json, pathlib, re, sys

BERTAA = pathlib.Path(__file__).parent

# (old_path, new_path) using raw strings (single backslash form).
# Order matters: more-specific paths must come before catch-all BERT_projects.
SINGLE = [
    (r'C:\Users\ivank\Documents\BERT_projects\BERT\results_45000',
     r'..\BertAA_content\Model\Checkpoints\results_45000'),
    (r'C:\Users\ivank\Documents\BERT_projects\BERT\results_42601',
     r'..\BertAA_content\Model\Checkpoints\results_42601'),
    (r'C:\Users\ivank\Documents\BERT_projects\BERT\results4',
     r'..\BertAA_content\Model\Checkpoints\results4'),
    (r'C:\Users\ivank\Documents\BERT_projects\BERT',
     r'..\BertAA_content\Model\Checkpoints'),
    (r'C:\Users\ivank\Documents\BERT_projects\results2',
     r'..\BertAA_content\Model\Checkpoints\results2'),
    (r'C:\Users\ivank\Documents\BERT_projects\results4',
     r'..\BertAA_content\Model\Checkpoints\results4'),
    (r'C:\Users\ivank\Documents\BERT_projects\Embeddings',
     r'..\BertAA_content\Model\Embeddings'),
    (r'C:\Users\ivank\Documents\BERT_projects\Classifier',
     r'..\BertAA_content\Model\Classifier'),
    (r'C:\Users\ivank\Documents\BERT_projects\Data',
     r'..\BertAA_content\Data'),
    (r'C:\Users\ivank\Documents\BERT_projects\100_examples',
     r'..\BertAA_content\Data\100_examples'),
    (r'C:\Users\ivank\Documents\BERT_projects\model_segm_torch',
     r'..\BertAA_content\Model\model_segm_torch'),
    (r'C:\Users\ivank\Documents\BERT_projects',
     r'..\BertAA_content'),
    (r'D:\pan20-authorship-verification-training-small',
     r'..\BertAA_content\Data'),
]

# In raw JSON text, each Python source backslash is escaped:
#   r"C:\Users"  in source  -> C:\\Users  in raw JSON  (double)
#   "C:\\Users"  in source  -> C:\\\\Users in raw JSON (quadruple)
DOUBLE    = [(o.replace('\\', '\\\\'),         n.replace('\\', '\\\\'))         for o, n in SINGLE]
QUADRUPLE = [(o.replace('\\', '\\\\\\\\'),     n.replace('\\', '\\\\\\\\'))     for o, n in SINGLE]

# A few cells in Final_model_PT.ipynb use a mixed form:
#   C:\\\\Users\\ivank\\\\Documents\\\\...  (4 backslashes before Users/Docs, 2 before ivank)
# Expressed as Python string values using '\\'*N notation for clarity:
_4 = '\\' * 4
_2 = '\\' * 2
_mixed_prefix = 'C:' + _4 + 'Users' + _2 + 'ivank' + _4 + 'Documents' + _4 + 'BERT_projects'
_mixed_new    = '..' + _4 + 'BertAA_content'
MIXED = [
    (_mixed_prefix + _4 + 'Embeddings', _mixed_new + _4 + 'Model' + _4 + 'Embeddings'),
    (_mixed_prefix,                      _mixed_new),
]

# Also handle pan20-large dataset path (appears without escaping in some cells)
PAN20_LARGE = [
    (r'D:\pan20-authorship-verification-training-large',
     r'..\BertAA_content\Data'),
]
PAN20_LARGE_DBL = [(o.replace('\\', '\\\\'), n.replace('\\', '\\\\')) for o, n in PAN20_LARGE]
PAN20_LARGE_QUAD = [(o.replace('\\', '\\\\\\\\'), n.replace('\\', '\\\\\\\\')) for o, n in PAN20_LARGE]


def patch_text(text):
    changes = []
    for label, pairs in [('SINGLE', SINGLE), ('DOUBLE', DOUBLE), ('QUADRUPLE', QUADRUPLE),
                         ('MIXED', MIXED),
                         ('PAN20L-S', PAN20_LARGE), ('PAN20L-D', PAN20_LARGE_DBL),
                         ('PAN20L-Q', PAN20_LARGE_QUAD)]:
        for old, new in pairs:
            if old in text:
                text = text.replace(old, new)
                changes.append(f"  {label}: ...{old[-30:]} -> ...{new[-30:]}")
    return text, changes


def patch_notebook(path):
    text = path.read_text(encoding='utf-8')
    new_text, changes = patch_text(text)
    if changes:
        path.write_text(new_text, encoding='utf-8')
        print(f"PATCHED {path.name}:")
        for c in changes:
            print(c)
    else:
        print(f"  clean  {path.name}")


def patch_pyfile(path):
    text = path.read_text(encoding='utf-8')
    new_text, changes = patch_text(text)
    if changes:
        path.write_text(new_text, encoding='utf-8')
        print(f"PATCHED {path.name}:")
        for c in changes:
            print(c)
    else:
        print(f"  clean  {path.name}")


if __name__ == '__main__':
    print("=== Patching notebooks ===")
    for nb in sorted(BERTAA.glob('**/*.ipynb')):
        patch_notebook(nb)

    print("\n=== Patching .py files ===")
    patch_pyfile(BERTAA / 'bertviz_test.py')

    print("\nDone.")
