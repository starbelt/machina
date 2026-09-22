"""
Reconcile the params CSV with what the code declares.

Adopted from icarus-dynamics ``params/sync.py``. The split that makes this
safe: **sync writes, check never does.** CI runs ``check``. A CI job that
mutates the repository destroys the only property worth having -- that a
green check means a human saw the diff.

Three things happen, and the third is the one with teeth:

* A declaration with no row gets one, with every human-owned column
  **blank**. The tool does not invent a number.
* Tool-owned columns are refreshed from the declaration. Human-owned columns
  are never touched, which is what lets a hand-set value survive a bounds
  change.
* A row with no declaration is **reported, not deleted**. Removal needs
  ``--prune``. A silent drop would remove a number something depends on.
"""

from dataclasses import dataclass, field
from pathlib import Path

from machina.params import table

__all__ = ["Diff", "plan", "apply", "sync"]


@dataclass
class Diff:
    """What ``sync`` would do, computed without doing it, so ``check`` can render it too."""

    added: list = field(default_factory=list)
    orphaned: list = field(default_factory=list)
    # name -> [(column, in_file, from_declaration)]
    retagged: dict = field(default_factory=dict)
    reordered: bool = False

    def __bool__(self) -> bool:
        return bool(self.added or self.orphaned or self.retagged or self.reordered)

    def render(self, *, pruned: bool = False) -> list:
        """``pruned`` says whether the caller is acting on the orphans or refusing to.

        Without it the orphan line would read "pass --prune" even on the run
        where --prune was passed and the rows were removed.
        """
        lines = []
        for name in self.added:
            lines.append(f"  {name}: declared in code, missing from the table")
        for name in self.orphaned:
            lines.append(
                f"  {name}: REMOVED from the table; it has no declaration" if pruned else
                f"  {name}: in the table with no declaration. Removing it could drop a number "
                f"something depends on, so sync leaves it alone -- pass --prune if that is "
                f"what you mean")
        for name, deltas in self.retagged.items():
            for column, got, want in deltas:
                lines.append(
                    f"  {name}.{column}: table says {got!r}, declaration says {want!r}. This "
                    f"column is owned by the declaration -- change it in Python, not here")
        if self.reordered:
            lines.append("  rows are not sorted by name")
        return lines


def _human(row) -> dict:
    return {c: row.get(c, "") for c in table.human_owned()}


def plan(decls: dict, rows: list) -> Diff:
    by_name = {r["name"]: r for r in rows}
    diff = Diff()
    for name, decl in decls.items():
        row = by_name.get(name)
        if row is None:
            diff.added.append(name)
            continue
        want = table.row_from_decl(decl, **_human(row))
        deltas = [(c, row[c], want[c]) for c in table.TOOL_OWNED if row[c] != want[c]]
        if deltas:
            diff.retagged[name] = deltas
    diff.orphaned = [r["name"] for r in rows if r["name"] not in decls]
    names = [r["name"] for r in rows]
    diff.reordered = names != sorted(names)
    return diff


def apply(decls: dict, rows: list, *, prune: bool = False) -> list:
    """The rows ``sync`` would write. Pure, so ``check`` can compare without a tempfile."""
    by_name = {r["name"]: r for r in rows}
    out = []
    for name, decl in decls.items():
        old = by_name.get(name)
        out.append(table.row_from_decl(decl, **(_human(old) if old else {})))
    if not prune:
        out.extend(r for r in rows if r["name"] not in decls)
    return sorted(out, key=lambda r: r["name"])


def sync(csv_path: Path, decls: dict, *, prune: bool = False) -> Diff:
    rows = table.read(csv_path)
    diff = plan(decls, rows)
    table.write(csv_path, apply(decls, rows, prune=prune))
    return diff
