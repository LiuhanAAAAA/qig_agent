# tools/_bootstrap.py
from __future__ import annotations

import os
import sys
from pathlib import Path


def add_project_root_to_syspath(marker: str = "src") -> str:
    """
    Make sure repo root is in sys.path so `import src.xxx` works
    no matter where the script is launched from.

    marker: a directory name that must exist at repo root (default: 'src')
    Returns: resolved root path
    """
    here = Path(__file__).resolve()
    root = here.parent.parent  # tools/.. -> repo root

    # sanity check: ensure root contains marker directory
    if not (root / marker).exists():
        # fallback: walk parents
        for p in here.parents:
            if (p / marker).exists():
                root = p
                break

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # also support editable style imports
    os.environ.setdefault("PYTHONPATH", root_str)
    return root_str
