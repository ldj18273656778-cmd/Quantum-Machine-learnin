"""Auto-configure import path for scripts executed from code/Train.

When running files like `python code/Train/yinyong_test.py`, Python puts
`code/Train` on `sys.path` but not necessarily `code`. This file is imported
automatically by Python's `site` module and adds `code` so `import sampling`
works in all Train scripts.
"""

from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
code_dir_str = str(CODE_DIR)
if code_dir_str not in sys.path:
    sys.path.insert(0, code_dir_str)
