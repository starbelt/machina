"""
``MANIFEST.json`` -- the index an exported artifact carries.

Adopted from icarus-dynamics ``codegen/manifest.py``. Every export stage writes
its own block and **merges** rather than replaces, so stages can run in any
order; ``sort_keys=True`` and a trailing newline make the bytes independent of
merge order, which the two-process determinism diff relies on.

There is deliberately no timestamp anywhere. A time in a hashed artifact makes
two otherwise identical builds differ; provenance is the commit hash instead.
"""

import json
import subprocess
from pathlib import Path

__all__ = ["merge", "git_commit", "canonical_json"]


def canonical_json(data) -> str:
    """The one spelling of a manifest: sorted keys, two-space indent, LF, trailing newline."""
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def merge(path: Path, block: dict) -> dict:
    """Add ``block`` to the manifest at ``path`` rather than replacing it.

    Top-level keys in ``block`` replace keys of the same name; everything else
    is kept. Returns the whole merged manifest. The parent directory is
    created if needed.
    """
    path = Path(path)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(existing, dict):
        raise ValueError(f"{path} is not a JSON object; refusing to merge into it.")
    existing.update(block)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(canonical_json(existing))
    return existing


def git_commit(repo: Path = None):
    """The commit an artifact was built from, or ``None`` outside a git tree."""
    if repo is None:
        return None
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None
