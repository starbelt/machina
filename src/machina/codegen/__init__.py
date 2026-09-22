"""
machina.codegen -- exporting built Functions as deterministic artifacts.

``manifest`` merges per-stage blocks into ``MANIFEST.json`` without
timestamps; ``export`` densifies, generates C, hashes files and describes
vector layouts. Nothing here knows about a particular model.
"""

from machina.codegen.export import dense, generate_c, layout_block, schema_hash, sha256_of
from machina.codegen.manifest import canonical_json, git_commit, merge

__all__ = ["dense", "generate_c", "sha256_of", "layout_block", "schema_hash",
           "merge", "git_commit", "canonical_json"]
