"""
Helpers for exporting CasADi Functions as artifacts.

icarus-dynamics keeps three of these behind a private spelling in
``codegen/plant_export.py``, with a note that "a third exporter is the point
at which they earn their own module". machina is that third exporter.

:func:`dense` is the non-obvious one. CasADi tracks structural sparsity, so a
Function whose output contains a structural zero -- a matrix built with
``ca.diag``, an unproduced ``SUM`` signal -- generates C that writes fewer
doubles than the output's dense length. A consumer that trusts the published
offsets then reads every later entry from the wrong slot, and nothing errors.
Densifying every output before code generation makes the written length equal
the published one.

:func:`generate_c` writes through ``ca.CodeGenerator`` into a temporary
directory, because ``Function.generate()`` writes relative to the working
directory and leaves a stray ``.c`` file in whatever repository ran it.
"""

import hashlib
import json
import tempfile
from pathlib import Path

import casadi as ca

__all__ = ["dense", "generate_c", "sha256_of", "layout_block", "schema_hash"]


def dense(f: ca.Function, name: str = None) -> ca.Function:
    """The same Function, optionally renamed, with every output dense."""
    sym = ca.MX.sym if f.is_a("MXFunction") else ca.SX.sym   # MX-only nodes (solve, rootfinder)
    args = [sym(f.name_in(i), f.size1_in(i), f.size2_in(i)) for i in range(f.n_in())]
    outs = list(f.call(args))
    return ca.Function(name or f.name(), args, [ca.densify(o) for o in outs],
                       [f.name_in(i) for i in range(f.n_in())],
                       [f.name_out(i) for i in range(f.n_out())])


def generate_c(f: ca.Function) -> str:
    """CasADi's C for one Function, as text. Leaves nothing behind on disk."""
    with tempfile.TemporaryDirectory() as tmp:
        generator = ca.CodeGenerator(f"{f.name()}.c", {"with_header": False})
        generator.add(f)
        written = generator.generate(str(Path(tmp)) + "/")
        return Path(written).read_text(encoding="utf-8")


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def layout_block(order, registry) -> dict:
    """``{path: {offset, shape, unit, frame}}`` for one vector, in vector order.

    ``order`` is a builder order (``state_order``, ``input_order``,
    ``algebraic_order``); ``registry`` is the signal registry the builder
    used. Instance paths are looked up by their last segment, which is the
    signal name. Offsets count elements, column-major, as the builder packs.
    """
    block, at = {}, 0
    for path in order:
        signal = registry.get(path.rpartition("/")[2])
        block[path] = {"offset": at, "shape": list(signal.shape), "unit": signal.unit,
                       "frame": signal.frame}
        at += signal.size
    return block


def schema_hash(block) -> int:
    """A u64 identifying a layout: the first 8 bytes of a canonical-JSON SHA-256,
    little-endian. Stamped into a binary log header, it ties a payload to the
    layout that wrote it."""
    canonical = json.dumps(block, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(canonical).digest()[:8], "little")
