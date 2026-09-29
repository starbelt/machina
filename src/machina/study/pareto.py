"""
Pareto frontiers over a table of evaluated configurations.

The rows come from outside the optimizer: a replay evaluator, a sweep, or a CSV
of operating points. Each row is one configuration and each objective is a
column. This module reads such a table and answers four questions:

- which rows are non-dominated (:func:`nondominated`), and which rows were not
  scored at all and why (:func:`excluded`: listed, not scored);
- where each row sits between the ideal and the nadir point (:func:`normalize`);
- for which objective weights each row is the weighted-sum optimum, and the
  exact weights at which that optimum switches from one row to the next
  (:func:`weighted_sweep`);
- the best row on one objective with the others held to bounds
  (:func:`epsilon_constraint`).

Tables
------
A table is any of: an object with an ``as_columns()`` method returning a
mapping of column -> sequence; a pandas DataFrame (recognised by its module
name, so this module never imports pandas); a mapping of column -> sequence;
or a sequence of mapping rows, whose columns are the union of the row keys in
first-seen order, a missing key reading as NaN. Every column has one value per
row. Objective columns must be numeric.

Objectives
----------
An :class:`Objective` names a column, its sense (``"min"`` or ``"max"``) and
whether to compare it on a natural-log axis. A plain string in an objectives
list means ``Objective(name)``, minimized on a linear axis. A row is excluded
(never on a front, NaN after normalization) when an objective is NaN or
infinite, when a log objective is not positive, or when the row is infeasible.
Dominance is computed on the raw values, sense-oriented: the log is monotone,
so it cannot change which rows dominate which. Normalization happens on the
logged values, where it does make a difference.

Normalization and scalarization
-------------------------------
``x̂ = (x - ideal) / (nadir - ideal)`` per objective, in the (logged) objective
space. The ideal is the best value over the reference rows; the nadir is the
worst value over the reference front (or over all reference rows). One formula
serves both senses: x̂ = 0 at the ideal and 1 at the nadir whether the
objective is minimized or maximized (for a maximized objective the
denominator is negative). Rows outside the reference set, such as an oracle
upper bound, can land outside [0, 1].

Downstream, and in :func:`weighted_sweep`, the scalarization is
``J_w = Σ_i w_i x̂_i`` with ``w_i >= 0`` and ``Σ w_i = 1``, minimized. Every
term is a normalized distance from the ideal, so a maximized objective enters
as x̂ itself. The form ``w·L̂ + (1 − w)(1 − η̂)`` is wrong with this
normalization: η̂ is already 0 at the best η, so ``1 − η̂`` is largest there
and that form penalizes the best η. Use η̂ directly.

Weighted-sum switch points
--------------------------
With two objectives the sweep is exact: the rows that win on some interval of
``w`` are the vertices of the lower convex hull of the normalized points, from
the minimum of objective 1 to the minimum of objective 2, and the switch
weight between neighbouring vertices is solved in closed form. Non-dominated
rows in a concave region of the front never win for any ``w``; they are
reported as ``unsupported``. With three or more objectives the sweep samples a
deterministic simplex grid of weights instead.

This module depends on numpy only.
"""

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Objective", "nondominated", "excluded", "normalize", "Normalized",
    "weighted_sweep", "WeightedSweep", "Segment", "epsilon_constraint",
]

_SENSES = ("min", "max")
_NADIR = ("front", "all")
# Elements per boolean block in the dominance test: bounds memory at a few MB.
_BLOCK = 1 << 22


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Objective:
    """
    One objective column.

    ``name`` is the column name. ``sense`` is ``"min"`` or ``"max"``. ``log``
    compares and normalizes the natural log of the column; rows where it is not
    positive are excluded.
    """

    name: str
    sense: str = "min"
    log: bool = False

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"Objective name must be a non-empty column name, got {self.name!r}")
        if not isinstance(self.sense, str) or self.sense not in _SENSES:
            raise ValueError(
                f"Objective {self.name!r}: sense must be 'min' or 'max', got {self.sense!r}")
        if not isinstance(self.log, bool):
            raise TypeError(
                f"Objective {self.name!r}: log must be True or False, got {self.log!r}")


@dataclass(frozen=True, eq=False)
class Normalized:
    """
    Normalized objective values; see :func:`normalize`.

    ``values`` is ``(n, m)``: 0 at the ideal, 1 at the nadir for every objective
    whatever its sense, NaN in excluded rows. ``ideal`` and ``nadir`` are ``(m,)``
    in the (logged) objective space. ``degenerate`` names the objectives whose
    ideal equals their nadir; their column is 0 in every valid row. The arrays
    are read-only; equality is identity.
    """

    values: np.ndarray
    ideal: np.ndarray
    nadir: np.ndarray
    objectives: tuple
    degenerate: tuple


@dataclass(frozen=True)
class Segment:
    """``rows`` (ascending) minimize ``J_w`` for every ``w`` in ``[w_lo, w_hi]``."""

    w_lo: float
    w_hi: float
    rows: tuple


@dataclass(frozen=True, eq=False)
class WeightedSweep:
    """
    Weighted-sum optima over the candidate rows; see :func:`weighted_sweep`.

    Two objectives: ``segments`` tile ``[0, 1]`` in ascending ``w`` and
    ``switch_points`` are the interior breakpoints. More objectives: ``grid``
    holds ``(weights, rows)`` per simplex grid point, and ``segments`` and
    ``switch_points`` are empty. All row tuples are ascending.
    """

    segments: tuple
    switch_points: tuple
    supported: tuple
    weakly_supported: tuple
    unsupported: tuple
    grid: tuple
    normalized: Normalized


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def nondominated(table, objectives, *, feasible=None) -> np.ndarray:
    """
    Boolean mask, one entry per row: True where the row is valid and no other
    valid row dominates it.

    Row j dominates row i when j is no worse on every objective and strictly
    better on at least one. Exact duplicates do not dominate each other, so all
    copies of a non-dominated point are kept. Invalid rows (see :func:`excluded`)
    are False.
    """
    prepared = _prepare(table, objectives, feasible)
    return _front(prepared.oriented, prepared.valid)


def excluded(table, objectives, *, feasible=None) -> list:
    """
    ``[(row, reason), ...]`` in row order for every row that is not scored.

    Reasons, joined with ``"; "`` when a row has several: ``"nan in 'L'"``,
    ``"inf in 'L'"``, ``"log of non-positive 'L'"`` (objective order), then
    ``"infeasible"``.
    """
    return list(_prepare(table, objectives, feasible).reasons)


def normalize(table, objectives, *, over=None, nadir="front", feasible=None) -> Normalized:
    """
    Normalize every valid row between the ideal (0) and the nadir (1).

    ``over`` selects the reference rows (a boolean mask or row indices; default
    all valid rows). The ideal is the best value over the valid reference rows.
    The nadir is the worst value over their non-dominated subset
    (``nadir="front"``) or over all of them (``nadir="all"``). Values are
    computed in the logged space for log objectives. Rows outside ``over`` are
    normalized with the same ideal and nadir and may fall outside [0, 1].
    """
    prepared = _prepare(table, objectives, feasible)
    return _normalize(prepared, over, nadir)


def weighted_sweep(table, objectives, *, over=None, among=None, nadir="front",
                   feasible=None, resolution=20, tol=1e-12) -> WeightedSweep:
    """
    Where the minimum of ``J_w = Σ_i w_i x̂_i`` sits as the weights vary.

    ``x̂`` is :func:`normalize` with ``over`` and ``nadir``. The candidates are
    the valid rows in ``among`` (a boolean mask or row indices; default
    ``over``, default all rows). Only candidates that are non-dominated among
    the candidates can win on an interval; dominated candidates appear in no
    field.

    Two objectives (exact): ``J_w = w·x̂_1 + (1 − w)·x̂_2`` for ``w`` in [0, 1].
    ``supported`` rows are the hull vertices, each the unique optimum (with its
    exact duplicates) on a segment of positive length unless a switch point
    rounds to 0 or 1. ``weakly_supported`` rows tie the envelope, within
    ``tol · max(1, |J|)``, at a breakpoint or an end of [0, 1] without being a
    vertex (collinear points). ``unsupported`` rows never reach the envelope.
    At ``w = 1`` the winner minimizes x̂_1 (ties to the smaller x̂_2); at
    ``w = 0`` it minimizes x̂_2 (ties to the smaller x̂_1). ``resolution`` is
    unused.

    Three or more objectives (sampled): every weight vector ``parts / resolution``
    with ``parts`` a composition of ``resolution`` into m non-negative integers,
    in lexicographic order; C(resolution + m − 1, m − 1) points. Each grid point
    records every candidate within ``tol · max(1, |min J|)`` of the minimum.
    ``supported`` is every row that wins somewhere on the grid;
    ``weakly_supported`` is empty; ``unsupported`` is the rest of the
    non-dominated candidates (a finer grid can move a row from unsupported to
    supported).
    """
    objs = _objectives(objectives)
    m = len(objs)
    if m < 2:
        raise ValueError(
            f"weighted_sweep needs at least two objectives, got {m} "
            f"({[o.name for o in objs]}); for one objective take the best row directly")
    if isinstance(resolution, bool) or not isinstance(resolution, numbers.Integral) \
            or resolution < 1:
        raise ValueError(f"weighted_sweep: resolution must be a positive int, got {resolution!r}")
    tol = float(tol)
    if not (np.isfinite(tol) and tol >= 0.0):
        raise ValueError(f"weighted_sweep: tol must be finite and >= 0, got {tol!r}")

    prepared = _prepare(table, objs, feasible)
    norm = _normalize(prepared, over, nadir)
    n = len(prepared.valid)
    over_mask = _row_mask(over, n, "over")
    among_mask = over_mask if among is None else _row_mask(among, n, "among")
    candidates = prepared.valid & among_mask
    if not candidates.any():
        raise ValueError(
            "weighted_sweep: no valid row among the candidates; check `among`/`over` "
            "and excluded(table, objectives) for the rows that were dropped")
    rows = np.flatnonzero(_front(prepared.oriented, candidates))
    points = norm.values[rows]
    if not np.isfinite(points).all():
        bad = [objs[i].name for i in range(m) if not np.isfinite(points[:, i]).all()]
        raise ValueError(
            f"weighted_sweep: normalized values overflow for {bad}; the ideal and nadir of "
            "those objectives are too close to divide by -- drop the objective or widen `over`")

    if m == 2:
        return _sweep_two(points, rows, tol, norm)
    return _sweep_grid(points, rows, int(resolution), tol, norm)


def epsilon_constraint(table, objectives, *, primary, bounds, feasible=None):
    """
    Best valid row on objective ``primary`` with every objective in ``bounds``
    no worse than its bound; ``None`` when no row qualifies.

    ``bounds`` maps objective name -> bound in the column's own units (not
    logged): ``x <= bound`` for a minimized objective, ``x >= bound`` for a
    maximized one. Ties on ``primary`` go to the lowest row index.
    """
    prepared = _prepare(table, objectives, feasible)
    names = [o.name for o in prepared.objectives]
    if primary not in names:
        raise ValueError(
            f"epsilon_constraint: primary {primary!r} is not one of the objectives {names}")
    if not isinstance(bounds, Mapping):
        raise TypeError(
            f"epsilon_constraint: bounds must be a mapping of objective name -> bound, "
            f"got {type(bounds).__name__}")
    signs = _signs(prepared.objectives)
    ok = prepared.valid.copy()
    for name, bound in bounds.items():
        if name not in names:
            raise ValueError(
                f"epsilon_constraint: bound on {name!r}, which is not one of the objectives "
                f"{names}; add it to the objectives or drop the bound")
        value = float(bound)
        if np.isnan(value):
            raise ValueError(f"epsilon_constraint: the bound on {name!r} is NaN")
        i = names.index(name)
        ok &= prepared.oriented[:, i] <= signs[i] * value
    qualified = np.flatnonzero(ok)
    if qualified.size == 0:
        return None
    column = prepared.oriented[qualified, names.index(primary)]
    return int(qualified[int(np.argmin(column))])


# ---------------------------------------------------------------------------
# Table and argument adapters
# ---------------------------------------------------------------------------

def _columns(table) -> dict:
    """``{column: 1-D array}`` from any supported table form; equal lengths checked."""
    as_columns = getattr(table, "as_columns", None)
    if callable(as_columns):
        source = as_columns()
        where = f"{type(table).__name__}.as_columns()"
        if not isinstance(source, Mapping):
            raise TypeError(
                f"{where} must return a mapping of column -> sequence, "
                f"got {type(source).__name__}")
        return _checked({key: _column(key, v, where) for key, v in source.items()}, where)
    if type(table).__module__.startswith("pandas"):
        if not hasattr(table, "columns"):
            raise TypeError(
                f"a pandas {type(table).__name__} is not a table; pass a DataFrame "
                "(Series.to_frame() makes one)")
        return _checked({c: _column(c, table[c].to_numpy(), "DataFrame")
                         for c in table.columns}, "DataFrame")
    if isinstance(table, Mapping):
        return _checked({key: _column(key, v, "mapping") for key, v in table.items()},
                        "mapping")
    if isinstance(table, Sequence) and not isinstance(table, (str, bytes)):
        return _from_rows(table)
    raise TypeError(
        f"cannot read a table from a {type(table).__name__}; pass a mapping of "
        "column -> sequence, a sequence of mapping rows, a pandas DataFrame, or an object "
        "with as_columns()")


def _from_rows(rows) -> dict:
    keys = {}  # insertion-ordered: first-seen column order
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(
                f"row {i} of the table is a {type(row).__name__}, not a mapping; a sequence "
                "table needs one mapping (column -> value) per row")
        for key in row:
            keys.setdefault(key, None)
    columns = {key: _column(key, [row[key] if key in row else np.nan for row in rows], "rows")
               for key in keys}
    return _checked(columns, "rows")


def _column(key, values, where) -> np.ndarray:
    """One column as a 1-D array; an object array when numpy cannot infer one."""
    try:
        array = np.asarray(values)
    except (ValueError, TypeError):
        array = None
    if array is not None and array.ndim == 1:
        return array
    if array is not None and array.ndim == 0:
        raise TypeError(
            f"column {key!r} of the {where} is a single value ({values!r}); "
            "give one value per row")
    items = list(values)
    out = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        out[i] = item
    return out


def _checked(columns, where) -> dict:
    lengths = {key: len(v) for key, v in columns.items()}
    if len(np.unique(list(lengths.values()))) > 1:
        raise ValueError(
            f"the columns of the {where} have different lengths {lengths}; "
            "every column needs one value per row")
    return columns


def _numeric(name, column) -> np.ndarray:
    kind = column.dtype.kind
    if kind in "biuf":
        return column.astype(float)
    if kind == "O":
        for i, value in enumerate(column):
            if value is not None and (isinstance(value, (str, bytes)) or not isinstance(
                    value, (numbers.Real, np.bool_))):
                raise TypeError(
                    f"objective column {name!r} holds the non-numeric value {value!r} at "
                    f"row {i}; objective columns must be numbers (NaN for missing)")
        return np.array([np.nan if v is None else float(v) for v in column], dtype=float)
    raise TypeError(
        f"objective column {name!r} is not numeric (dtype {column.dtype}); convert it to "
        "numbers or pick another column")


def _truthy(values, what) -> np.ndarray:
    kind = values.dtype.kind
    if kind == "b":
        return values.astype(bool)
    if kind in "iuf":
        if kind == "f" and np.isnan(values).any():
            row = int(np.flatnonzero(np.isnan(values))[0])
            raise ValueError(f"{what} is NaN at row {row}; give True or False for every row")
        return values != 0
    if kind == "O":
        out = np.empty(len(values), dtype=bool)
        for i, value in enumerate(values):
            if value is None or isinstance(value, (str, bytes)) or (
                    isinstance(value, float) and np.isnan(value)):
                raise ValueError(
                    f"{what} holds {value!r} at row {i}; give True or False for every row")
            out[i] = bool(value)
        return out
    raise TypeError(f"{what} is not boolean (dtype {values.dtype}); give True or False per row")


def _row_mask(spec, n, what) -> np.ndarray:
    """``None`` -> all rows; a boolean mask of length n; or a sequence of row indices."""
    if spec is None:
        return np.ones(n, dtype=bool)
    array = np.asarray(spec)
    if array.ndim != 1:
        raise ValueError(f"`{what}` must be a 1-D boolean mask or a sequence of row indices")
    if array.dtype.kind == "b":
        if len(array) != n:
            raise ValueError(
                f"`{what}` is a boolean mask of length {len(array)} but the table has {n} rows")
        return array.copy()
    mask = np.zeros(n, dtype=bool)
    if array.size == 0:
        return mask
    if array.dtype.kind not in "iu":
        raise TypeError(
            f"`{what}` must be a boolean mask or a sequence of int row indices, "
            f"got dtype {array.dtype}")
    if array.min() < 0 or array.max() >= n:
        raise IndexError(
            f"`{what}` holds row indices outside 0..{n - 1}: "
            f"{[int(i) for i in array if i < 0 or i >= n]}")
    mask[array] = True
    return mask


def _objectives(objectives) -> tuple:
    if isinstance(objectives, (str, Objective)):
        raise TypeError(
            f"objectives must be a sequence of Objective or column names; got the single "
            f"{objectives!r} -- wrap it in a list")
    try:
        items = list(objectives)
    except TypeError:
        raise TypeError(
            f"objectives must be a sequence of Objective or column names, "
            f"got {type(objectives).__name__}") from None
    out = []
    for item in items:
        if isinstance(item, str):
            item = Objective(item)
        elif not isinstance(item, Objective):
            raise TypeError(
                f"objectives must be Objective instances or column names, got {item!r}")
        out.append(item)
    if not out:
        raise ValueError("objectives is empty; name at least one objective column")
    names = [o.name for o in out]
    for i, name in enumerate(names):
        if name in names[:i]:
            raise ValueError(f"objective {name!r} is listed twice in {names}; list it once")
    return tuple(out)


def _signs(objectives) -> np.ndarray:
    return np.array([1.0 if o.sense == "min" else -1.0 for o in objectives])


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class _Prepared:
    objectives: tuple
    logged: np.ndarray    # (n, m), log applied where requested; NaN in invalid rows
    oriented: np.ndarray  # (n, m) raw values times the sense sign: smaller is better
    valid: np.ndarray     # (n,)
    reasons: tuple        # ((row, reason), ...)


def _prepare(table, objectives, feasible) -> _Prepared:
    objs = _objectives(objectives)
    columns = _columns(table)
    available = list(columns)
    for o in objs:
        if o.name not in columns:
            raise KeyError(
                f"objective {o.name!r} is not a column of the table; available: {available}")
    raw = np.stack([_numeric(o.name, columns[o.name]) for o in objs], axis=1)
    n, m = raw.shape

    if feasible is None:
        feasible_mask = np.ones(n, dtype=bool)
    elif isinstance(feasible, str):
        if feasible not in columns:
            raise KeyError(
                f"feasible column {feasible!r} is not a column of the table; "
                f"available: {available}")
        feasible_mask = _truthy(columns[feasible], f"feasible column {feasible!r}")
    else:
        values = _column("feasible", feasible, "feasible argument")
        if len(values) != n:
            raise ValueError(
                f"feasible has {len(values)} entries but the table has {n} rows; "
                "give one per row or name a column")
        feasible_mask = _truthy(values, "feasible")

    nan = np.isnan(raw)
    inf = np.isinf(raw)
    wants_log = np.array([o.log for o in objs], dtype=bool)
    nonpositive = wants_log[None, :] & ~nan & ~inf & (raw <= 0.0)
    bad = nan | inf | nonpositive
    valid = ~bad.any(axis=1) & feasible_mask

    reasons = []
    for row in np.flatnonzero(~valid):
        parts = []
        for i, o in enumerate(objs):
            if nan[row, i]:
                parts.append(f"nan in {o.name!r}")
            elif inf[row, i]:
                parts.append(f"inf in {o.name!r}")
            elif nonpositive[row, i]:
                parts.append(f"log of non-positive {o.name!r}")
        if not feasible_mask[row]:
            parts.append("infeasible")
        reasons.append((int(row), "; ".join(parts)))

    logged = np.full((n, m), np.nan)
    for i, o in enumerate(objs):
        column = raw[valid, i]
        logged[valid, i] = np.log(column) if o.log else column
    oriented = raw * _signs(objs)[None, :]
    oriented[~valid] = np.nan
    return _Prepared(objs, logged, oriented, valid, tuple(reasons))


def _dominated(z) -> np.ndarray:
    """``(k,)`` True where some other row of ``z`` (``(k, m)``, finite, minimize) dominates."""
    k, m = z.shape
    out = np.zeros(k, dtype=bool)
    if k == 0:
        return out
    step = max(1, _BLOCK // max(1, k * m))
    for start in range(0, k, step):
        block = z[start:start + step][None, :, :]  # (1, B, m): the rows tested
        others = z[:, None, :]                     # (k, 1, m): the rows that might dominate
        no_worse = (others <= block).all(axis=2)
        better = (others < block).any(axis=2)
        out[start:start + step] = (no_worse & better).any(axis=0)
    return out


def _front(oriented, mask) -> np.ndarray:
    """Rows in ``mask`` that no other row in ``mask`` dominates; full-length mask."""
    rows = np.flatnonzero(mask)
    out = np.zeros(len(mask), dtype=bool)
    out[rows] = ~_dominated(oriented[rows])
    return out


def _normalize(prepared, over, nadir) -> Normalized:
    if nadir not in _NADIR:
        raise ValueError(f"nadir must be 'front' or 'all', got {nadir!r}")
    objs = prepared.objectives
    n, m = prepared.logged.shape
    reference = prepared.valid & _row_mask(over, n, "over")
    if not reference.any():
        raise ValueError(
            "normalize: no valid row in `over`; excluded(table, objectives) lists the rows "
            "that were dropped and why")
    signs = _signs(objs)
    best_first = prepared.logged * signs[None, :]
    ideal = best_first[reference].min(axis=0) * signs
    basis = _front(prepared.oriented, reference) if nadir == "front" else reference
    worst = best_first[basis].max(axis=0) * signs

    values = np.full((n, m), np.nan)
    degenerate = []
    for i, o in enumerate(objs):
        span = worst[i] - ideal[i]
        if span == 0.0:
            values[prepared.valid, i] = 0.0
            degenerate.append(o.name)
        else:
            # "+ 0.0" turns the -0.0 of a maximized objective's ideal row into 0.0.
            values[prepared.valid, i] = (
                (prepared.logged[prepared.valid, i] - ideal[i]) / span + 0.0)
    for array in (values, ideal, worst):
        array.flags.writeable = False
    return Normalized(values, ideal, worst, objs, tuple(degenerate))


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _switch_weights(hull) -> list:
    """w*_j between hull[j] and hull[j + 1]: equal J there, hull[j] better above it."""
    out = []
    for j in range(len(hull) - 1):
        da = hull[j + 1][0] - hull[j][0]
        db = hull[j][1] - hull[j + 1][1]
        out.append(db / (db + da))
    return out


def _sweep_two(points, rows, tol, norm) -> WeightedSweep:
    # Unique points sorted by (a, b), exact duplicates grouped; rows stay ascending
    # within a group because lexsort is stable and ``rows`` is ascending.
    unique = []
    for idx in np.lexsort((points[:, 1], points[:, 0])):
        a, b = float(points[idx, 0]), float(points[idx, 1])
        if unique and unique[-1][0] == a and unique[-1][1] == b:
            unique[-1][2].append(int(rows[idx]))
        else:
            unique.append((a, b, [int(rows[idx])]))

    # Andrew's monotone chain, lower hull; cross <= 0 also drops collinear points.
    hull = []
    for p in unique:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) <= 0.0:
            hull.pop()
        hull.append(p)
    # Keep the chain from the minimum of a to the first vertex with the minimum b.
    b_min = min(v[1] for v in hull)
    hull = hull[:next(j for j, v in enumerate(hull) if v[1] == b_min) + 1]

    # Along the chain a rises and b falls, and w*_j falls: a larger w favours a
    # smaller a, so hull[0] wins at w = 1 and hull[-1] at w = 0. Rounding can break
    # the strict decrease for nearly collinear vertices; drop the middle vertex
    # there (it is then tested as weakly supported below).
    weights = _switch_weights(hull)
    j = 1
    while j < len(weights):
        if weights[j] < weights[j - 1]:
            j += 1
            continue
        del hull[j]
        weights = _switch_weights(hull)
        j = 1

    switch_points = tuple(reversed(weights))
    bounds = (0.0,) + switch_points + (1.0,)
    ordered = list(reversed(hull))  # ascending w
    segments = tuple(Segment(bounds[i], bounds[i + 1], tuple(v[2]))
                     for i, v in enumerate(ordered))

    supported_rows = sorted(r for v in hull for r in v[2])
    is_supported = np.isin(rows, supported_rows)
    ties = np.zeros(len(rows), dtype=bool)
    for w in bounds:
        value = w * points[:, 0] + (1.0 - w) * points[:, 1]
        envelope = value.min()
        ties |= np.abs(value - envelope) <= tol * max(1.0, abs(envelope))
    weak = ties & ~is_supported
    return WeightedSweep(
        segments=segments,
        switch_points=switch_points,
        supported=tuple(supported_rows),
        weakly_supported=tuple(int(r) for r in rows[weak]),
        unsupported=tuple(int(r) for r in rows[~is_supported & ~weak]),
        grid=(),
        normalized=norm,
    )


def _compositions(total, parts):
    """Compositions of ``total`` into ``parts`` non-negative ints, lexicographic order."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def _sweep_grid(points, rows, resolution, tol, norm) -> WeightedSweep:
    m = points.shape[1]
    wins = np.zeros(len(rows), dtype=bool)
    grid = []
    for parts in _compositions(resolution, m):
        weights = tuple(p / resolution for p in parts)
        value = np.zeros(len(rows))
        for i, w in enumerate(weights):  # fixed summation order: objective order
            value = value + w * points[:, i]
        envelope = value.min()
        tied = np.abs(value - envelope) <= tol * max(1.0, abs(envelope))
        wins |= tied
        grid.append((weights, tuple(int(r) for r in rows[tied])))
    return WeightedSweep(
        segments=(),
        switch_points=(),
        supported=tuple(int(r) for r in rows[wins]),
        weakly_supported=(),
        unsupported=tuple(int(r) for r in rows[~wins]),
        grid=tuple(grid),
        normalized=norm,
    )
