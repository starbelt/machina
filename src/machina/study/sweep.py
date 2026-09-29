"""
``sweep`` -- one solve per point of a grid, one row per point.

A design study asks one question at many points: the optimum as a power cap
moves, the coverage with the orbit pinned at each of twenty altitudes, a
component variant rebuilt per row. :func:`sweep` walks the cartesian grid of
its axes in axis order (the last axis varies fastest), solves each point, and
returns a :class:`SweepTable`: the axis values first, then how the solve
went, then whatever ``collect`` asked for.

Targets
-------
A built :class:`~machina.compiler.Problem`
    Each axis names an instance path and is routed by the role the path
    compiled as. ``PARAMETER`` reaches ``solve(values={path: v})`` for that
    call only: nothing is stored and nothing is rebuilt. ``VARIABLE`` or
    ``DISCRETE`` is pinned with ``fix(path, v)`` and started there with
    ``solve(x0={path: v})`` (a warm start's primal values would otherwise put
    it at the previous point's value, outside its new bounds). The bounds and
    initial guess such a variable had before the sweep are restored when the
    sweep ends, however it ends, so the Problem is left as it was found. A
    ``FIXED`` quantity was folded into the NLP as a constant and cannot be
    swept on this Problem.
A callable
    ``target(**point)`` returns a compiled Problem per point, so an axis can
    change structure: a factory parameter baked into a component, a component
    count. Axis names are its keyword names. ``sweep`` builds the Problem when
    the target did not.

Warm starts
-----------
With ``warm_start=True`` a point starts from the previous point's result when
that point succeeded, and cold when it did not: a failed solve never seeds the
next point. On a Problem target this is the backend's full warm start (primal
values, and duals when the layouts match). Across the rebuilds of a callable
target the result comes from another Problem; the backend copies primal values
by variable name where the element count matches and takes duals only when the
whole layout is identical, so a changed layout costs the duals, not the solve.
The one case the backend refuses -- a variable that kept its name and element
count but changed shape, e.g. (2, 3) -> (3, 2) -- is checked first, and such a
point starts cold. With ``retry_cold=True`` a warm-started point whose solve
fails is solved again from the Problem's stored initial guess; the row records
``retried=True`` and the retry's outcome.

Rows
----
Columns, in order: the axis names; then

``success``       the solver reported success (``SolutionResult.success``);
``status``        ``SolutionResult.status``, or ``"error"`` when the point raised;
``objective``     ``f_opt`` when the solve succeeded, NaN otherwise -- a Pareto
                  front over the table never picks up an unconverged iterate
                  (``keep_results=True`` keeps the raw result);
``iterations``    the solver's iteration count, ``None`` when the point raised;
``warm_started``  the recorded solve was seeded with the previous point's result
                  (``SolutionResult.warm_started`` is narrower: the plugin's
                  warm-start solver ran, which needs identical layouts and
                  finite duals);
``retried``       a warm-started solve failed and a cold one was run;
``error``         ``None``, or ``"<Type>: <first line of the message>"`` when the
                  point raised;

then the ``collect`` columns. ``t_wall`` (seconds inside the recorded solver
call, ``None`` when the point raised) is in every row dict but not in
``columns``: it differs between two runs of the same sweep, and the CSV and the
column view stay byte-stable without it unless asked (``include_timing=True``).

``collect`` entries are paths :meth:`Problem.evaluate` reports (quantities and
algebraic signals) or ``"cost:<name>"``, the cost term ``name`` as it enters
the objective (``SolutionResult.cost_terms``: weight applied). A scalar keeps
its name as the column; a vector or matrix expands to ``path[0]``,
``path[1]``, ... in column-major order. A point that did not succeed has NaN in
every collected column. Entries are checked before the first solve: against the
Problem target, or against the first Problem a callable target returns (whose
shapes then fix the columns for every later point).

Every point runs under ``try/except Exception``: an exception becomes a row with
``success=False``, ``status="error"`` and the message, and the sweep moves on;
``raise_on_error=True`` re-raises it instead. A target that returns something
other than a compiled Problem, and a bad ``collect`` entry, are refused outright,
whatever ``raise_on_error`` says: every point would hit them.

This module imports pandas only inside :meth:`SweepTable.to_pandas`.
"""

import csv
import itertools
import keyword
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from machina.compiler.problem import Problem
from machina.model.component import Role

__all__ = ["sweep", "SweepTable"]

STATUS_COLUMNS = ("success", "status", "objective", "iterations", "warm_started", "retried",
                  "error")
TIMING_COLUMN = "t_wall"
COST_PREFIX = "cost:"

_SWEEPABLE = (Role.PARAMETER, Role.VARIABLE, Role.DISCRETE)


# --- the table --------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SweepTable:
    """One row per grid point, in grid order.

    ``axes`` are the axis names, ``columns`` every column name in order (axes,
    :data:`STATUS_COLUMNS`, collected columns; ``t_wall`` is not one of them),
    ``rows`` one dict per point -- read-only by convention -- and ``results``
    the :class:`~machina.solver.SolutionResult` per row when the sweep ran with
    ``keep_results=True`` (``None`` for a point that raised), else ``None``.
    """

    axes: tuple
    columns: tuple
    rows: tuple
    results: tuple | None = None

    def __post_init__(self):
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "rows", tuple(self.rows))
        if self.results is not None:
            results = tuple(self.results)
            if len(results) != len(self.rows):
                raise ValueError(
                    f"SweepTable(results=...) has {len(results)} entries for {len(self.rows)} "
                    f"rows; give one result (or None) per row, or results=None."
                )
            object.__setattr__(self, "results", results)

    @classmethod
    def from_rows(cls, axes, rows) -> "SweepTable":
        """A table from plain dict rows, without solving anything.

        ``axes`` is a sequence of column names that every row carries.
        ``columns`` is the axes, then every other key in first-seen order;
        ``t_wall`` is kept in the rows but is not a column. Every row must hold
        every column (``None`` is an empty cell), and every value must be a
        scalar: ``None``, a bool, a number or a string.
        """
        if isinstance(axes, (str, bytes)) or isinstance(axes, Mapping):
            raise TypeError(
                f"SweepTable.from_rows(axes=...) takes a sequence of column names, got "
                f"{axes!r}; write axes=({axes!r},) for a single axis."
            )
        axes = tuple(axes)
        for name in axes:
            if not isinstance(name, str) or not name:
                raise TypeError(f"SweepTable.from_rows: axis names are strings, got {name!r}.")
        duplicated = [name for index, name in enumerate(axes) if name in axes[:index]]
        if duplicated:
            raise ValueError(f"SweepTable.from_rows: axes {duplicated} are given twice.")
        if isinstance(rows, (Mapping, str, bytes)):
            raise TypeError(
                "SweepTable.from_rows(rows=...) takes a sequence of dict rows, one per grid "
                "point; wrap a single row in a list."
            )
        rows = list(rows)
        columns = list(axes)
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TypeError(
                    f"SweepTable.from_rows: row {index} is a {type(row).__name__}, not a dict "
                    f"{{column: value}}."
                )
            for key in row:
                if not isinstance(key, str) or not key:
                    raise TypeError(
                        f"SweepTable.from_rows: row {index} has the key {key!r}; column names "
                        f"are strings."
                    )
                if key != TIMING_COLUMN and key not in columns:
                    columns.append(key)
        plain = []
        for index, row in enumerate(rows):
            missing = [column for column in columns if column not in row]
            if missing:
                raise ValueError(
                    f"SweepTable.from_rows: row {index} has no {missing}. Every row needs every "
                    f"column {columns}; write None for an empty cell."
                )
            keys = columns + ([TIMING_COLUMN] if TIMING_COLUMN in row else [])
            plain.append({key: _scalar(row[key], f"SweepTable.from_rows: row {index}, "
                                                 f"column {key!r}", allow_none=True)
                          for key in keys})
        return cls(axes=axes, columns=tuple(columns), rows=tuple(plain))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, column: str) -> list:
        """The values of one column, in row order. ``t_wall`` works too when every row has it."""
        if not isinstance(column, str):
            raise TypeError(
                f"SweepTable[...] takes a column name, got {column!r}; use table.rows[i] for "
                f"row i."
            )
        timing = column == TIMING_COLUMN and all(TIMING_COLUMN in row for row in self.rows)
        if column not in self.columns and not timing:
            raise KeyError(
                f"{column!r} is not a column of this SweepTable. Columns: {list(self.columns)}."
            )
        return [row[column] for row in self.rows]

    def as_columns(self, *, include_timing=False) -> dict:
        """``{column: list of values}`` in column order -- what :mod:`machina.study.pareto` reads.

        ``include_timing=True`` appends ``t_wall`` (``None`` in a row without it).
        """
        return {column: [row.get(column) for row in self.rows]
                for column in self._names(include_timing)}

    def to_csv(self, path, *, include_timing=False) -> None:
        """Write the table as UTF-8 CSV, byte-identical for identical rows on any interpreter.

        Header = ``columns`` (then ``t_wall`` with ``include_timing=True``);
        ``\\n`` line ends; floats as ``repr(float(x))`` (``nan``, ``inf``);
        bools ``True``/``False``; ints as themselves; ``None`` an empty cell;
        strings as they are, quoted by the csv module only where they hold a
        comma, a quote or a newline.
        """
        names = self._names(include_timing)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(names)
            for index, row in enumerate(self.rows):
                writer.writerow([_cell(row.get(name), name, index) for name in names])

    def to_pandas(self, *, include_timing=False):
        """The table as a pandas DataFrame, columns in order. pandas is imported here only."""
        try:
            import pandas
        except ImportError as exc:
            raise ImportError(
                "SweepTable.to_pandas() needs pandas, which machina keeps optional: install the "
                "study extra (pip install 'machina[study]'), or read the table without it via "
                "as_columns() or to_csv()."
            ) from exc
        names = self._names(include_timing)
        return pandas.DataFrame(self.as_columns(include_timing=include_timing), columns=names)

    def _names(self, include_timing) -> list:
        return list(self.columns) + ([TIMING_COLUMN] if include_timing else [])

    def __repr__(self) -> str:
        return (f"<SweepTable {len(self.rows)} row(s) x {len(self.columns)} column(s), "
                f"axes {list(self.axes)}>")


# --- sweep ------------------------------------------------------------------------------------


def sweep(target, axes, *, warm_start=True, retry_cold=True, collect=(), keep_results=False,
          raise_on_error=False) -> SweepTable:
    """Solve ``target`` at every point of the cartesian grid of ``axes``; one row per point.

    Args:
        target:         a built :class:`~machina.compiler.Problem`, or a callable
                        ``target(**point) -> Problem`` (compiled; built here if not).
        axes:           an ordered mapping ``{name: values}`` or a list of
                        ``(name, values)`` pairs. ``values`` is a list, tuple,
                        range or 1-D array of scalars (numbers, bools; strings for
                        a callable target). The grid is ``itertools.product`` in
                        axis order: the last axis varies fastest.
        warm_start:     seed each point with the previous point's result when that
                        point succeeded.
        retry_cold:     solve a warm-started point that failed again, cold.
        collect:        extra columns: ``Problem.evaluate`` paths, or
                        ``"cost:<name>"`` for a cost term with its weight applied.
        keep_results:   keep every ``SolutionResult`` in ``SweepTable.results``.
        raise_on_error: re-raise the first exception a point raises instead of
                        recording it.

    Returns:
        :class:`SweepTable`. The module docstring lists the columns.
    """
    grid = _axes(axes)
    names = tuple(name for name, _ in grid)
    collector = _Collector(collect, names)
    if isinstance(target, Problem):
        driver = _ProblemDriver(target, grid)
        collector.bind(target)
    elif callable(target):
        driver = _CallableDriver(target, grid)
    else:
        raise TypeError(
            f"sweep(target) takes a built Problem or a callable target(**point) that returns "
            f"one, got {type(target).__name__}."
        )

    records = []
    previous = None
    try:
        for combination in itertools.product(*(values for _, values in grid)):
            point = dict(zip(names, combination))
            record = _solve_point(driver, collector, point, previous if warm_start else None,
                                  retry_cold=retry_cold, raise_on_error=raise_on_error)
            records.append(record)
            result = record["result"]
            previous = result if result is not None and result.success else None
    finally:
        driver.restore()

    collected = collector.columns()
    rows = []
    for record in records:
        row = dict(record["point"])
        row.update(record["status"])
        values = record["collected"]
        for column in collected:
            row[column] = math.nan if values is None else values[column]
        row[TIMING_COLUMN] = record["t_wall"]
        rows.append(row)
    return SweepTable(
        axes=names, columns=names + STATUS_COLUMNS + collected, rows=tuple(rows),
        results=tuple(record["result"] for record in records) if keep_results else None)


def _solve_point(driver, collector, point, previous, *, retry_cold, raise_on_error) -> dict:
    """One grid point: make the Problem, solve (warm, then cold if that failed), collect."""
    record = {"point": point, "result": None, "collected": None, "t_wall": None}
    warm = False
    try:
        made = driver.make(point)
    except Exception as exc:
        if raise_on_error:
            raise
        return _error(record, exc, warm, retried=False)
    problem = driver.accept(made, point)            # refusals every point would hit: raise
    try:
        solve = driver.prepare(problem, point)
    except Exception as exc:
        if raise_on_error:
            raise
        return _error(record, exc, warm, retried=False)
    if not collector.bound:
        collector.bind(problem)                     # before the first solve, and raises

    retried = False
    try:
        seed = previous if previous is not None and driver.seedable(problem, previous) else None
        warm = seed is not None
        result = solve(seed)
        if warm and retry_cold and not result.success:
            warm, retried = False, True
            result = solve(None)
        collected = collector.values(problem, result) if result.success else None
    except Exception as exc:
        if raise_on_error:
            raise
        return _error(record, exc, warm, retried=retried)

    record.update(result=result, collected=collected, t_wall=_optional_float(result.t_wall))
    record["status"] = {
        "success": bool(result.success),
        "status": str(result.status),
        "objective": float(result.f_opt) if result.success else math.nan,
        "iterations": None if result.iterations is None else int(result.iterations),
        "warm_started": warm,
        "retried": retried,
        "error": None,
    }
    return record


def _error(record, exc, warm, *, retried) -> dict:
    record["status"] = {
        "success": False, "status": "error", "objective": math.nan, "iterations": None,
        "warm_started": warm, "retried": retried, "error": _describe(exc),
    }
    return record


# --- targets ----------------------------------------------------------------------------------


class _ProblemDriver:
    """A built Problem: parameters per call, variables pinned and restored afterwards."""

    def __init__(self, problem, grid):
        if not problem.is_compiled:
            raise RuntimeError(
                "sweep(target): the Problem is not compiled. Call problem.compile(...).build() "
                "first, or pass a callable target that returns a Problem per point."
            )
        if not problem.is_built:
            raise RuntimeError(
                "sweep(target): the Problem is compiled but not built. Call problem.build() "
                "first; sweep solves it without a rebuild per point."
            )
        roles = problem.roles
        free = [path for path, role in roles.items() if role in _SWEEPABLE]
        self._problem = problem
        self._parameters, self._variables = [], []
        for name, values in grid:
            role = roles.get(name)
            if role is None:
                raise ValueError(
                    f"sweep axis {name!r} is not a quantity of this Problem. Paths it can sweep "
                    f"(parameters and variables): {free or '(none)'}. A signal is computed, "
                    f"not set; to vary anything else, pass a callable target that builds a "
                    f"Problem per point."
                )
            if role is Role.FIXED:
                raise ValueError(
                    f"sweep axis {name!r} compiled as 'fixed': it is folded into the NLP as a "
                    f"constant, so no solve can change it. Build the Problem with "
                    f"compile(roles={{{name!r}: 'parameter'}}) to sweep it without a rebuild, "
                    f"or pass a callable target that builds a Problem per point."
                )
            for value in values:
                if isinstance(value, str):
                    raise TypeError(
                        f"sweep axis {name!r} routes to a {role.value} of the Problem, and "
                        f"{value!r} is not a number. Strings only make sense as keywords of a "
                        f"callable target."
                    )
            (self._parameters if role is Role.PARAMETER else self._variables).append(name)
        backend = problem.backend
        self._saved = {}
        for path in self._variables:
            lb, ub = backend.bounds(path)
            self._saved[path] = (np.array(lb, dtype=float), np.array(ub, dtype=float),
                                 np.array(backend.initial_guess(path), dtype=float))

    def make(self, point):
        return self._problem

    def accept(self, made, point):
        return made

    def prepare(self, problem, point):
        for path in self._variables:
            problem.fix(path, point[path])
        values = {path: point[path] for path in self._parameters} or None
        x0 = {path: point[path] for path in self._variables} or None

        def solve(seed):
            return problem.solve(values=values, x0=x0, warm_start=seed)
        return solve

    def seedable(self, problem, result) -> bool:
        return True

    def restore(self) -> None:
        """Bounds and initial guess of every swept variable, as they were before the sweep."""
        backend = self._problem.backend
        for path, (lb, ub, x0) in self._saved.items():
            self._problem.set_bounds(path, lb=lb, ub=ub)
            backend.set_initial_guess(path, x0)


class _CallableDriver:
    """``target(**point)`` per point; the Problem it returns is built here if it is not."""

    def __init__(self, target, grid):
        self._target = target
        self._label = getattr(target, "__name__", type(target).__name__)
        for name, _ in grid:
            if not name.isidentifier() or keyword.iskeyword(name):
                raise ValueError(
                    f"sweep axis {name!r} is not a Python identifier, so {self._label}(**point) "
                    f"cannot take it as a keyword. Name the axis with an identifier and route it "
                    f"inside the target, or sweep the path on a built Problem: "
                    f"sweep(problem, {{{name!r}: ...}})."
                )

    def make(self, point):
        return self._target(**point)

    def accept(self, made, point):
        if not isinstance(made, Problem):
            raise TypeError(
                f"sweep(target): {self._label}(**{point!r}) returned a {type(made).__name__}, "
                f"not a Problem. The target returns one compiled Problem per point: "
                f"return Problem(...).compile(...), and sweep builds it."
            )
        if not made.is_compiled:
            raise RuntimeError(
                f"sweep(target): {self._label}(**{point!r}) returned a Problem that is not "
                f"compiled. Compile it inside the target -- return Problem(...).compile(...) -- "
                f"and sweep builds it."
            )
        return made

    def prepare(self, problem, point):
        if not problem.is_built:
            problem.build()

        def solve(seed):
            return problem.solve(warm_start=seed)
        return solve

    def seedable(self, problem, result) -> bool:
        """False when a variable kept its name and element count but changed shape.

        The backend copies a previous result's primal values by name wherever the
        element count matches, and cannot lay a (2, 3) value into a (3, 2) variable.
        """
        backend = problem.backend
        names = backend.variable_order()
        for entry in result.x_layout:
            if entry.name not in names:
                continue
            shape = tuple(backend.shape_of(entry.name))
            before = tuple(entry.shape)
            if before != shape and before[0] * before[1] == shape[0] * shape[1]:
                return False
        return True

    def restore(self) -> None:
        pass


# --- collect ----------------------------------------------------------------------------------


class _Collector:
    """``collect`` entries -> columns, fixed by the first Problem they are checked against."""

    def __init__(self, collect, axis_names):
        if isinstance(collect, (str, bytes)):
            raise TypeError(
                f"sweep(collect={collect!r}) is a string, not a sequence of entries; did you "
                f"mean collect=({collect!r},)?"
            )
        if isinstance(collect, (set, frozenset)):
            raise TypeError(
                "sweep(collect=...) is a set, which has no order, so the column order would "
                "change from run to run. Pass a list or a tuple."
            )
        entries = list(collect)
        for entry in entries:
            if not isinstance(entry, str) or not entry:
                raise TypeError(
                    f"sweep(collect=...): {entry!r} is not an entry; give a path of the Problem "
                    f"or 'cost:<name>'."
                )
        duplicated = [entry for index, entry in enumerate(entries) if entry in entries[:index]]
        if duplicated:
            raise ValueError(f"sweep(collect=...) names {duplicated} twice; list each once.")
        self._entries = tuple(entries)
        self._reserved = tuple(axis_names) + STATUS_COLUMNS + (TIMING_COLUMN,)
        self._check_names(self._entries)
        self._layout = None

    @property
    def bound(self) -> bool:
        return self._layout is not None

    def bind(self, problem) -> None:
        """Check every entry against ``problem`` and fix the columns from its shapes."""
        builder = problem.builder
        paths = list(builder.quantity_order)
        paths += [path for path in builder.algebraic_order if path not in paths]
        costs = [record.name for record in problem.backend.cost_terms()]
        layout = []
        for entry in self._entries:
            if entry.startswith(COST_PREFIX):
                name = entry[len(COST_PREFIX):]
                if name not in costs:
                    raise ValueError(
                        f"sweep(collect=...): {entry!r} names no cost term of this Problem. "
                        f"Cost terms: {[COST_PREFIX + cost for cost in costs] or '(none)'} "
                        f"(collected weighted, as they enter the objective)."
                    )
                layout.append((entry, (entry,)))
                continue
            if entry not in paths:
                raise ValueError(
                    f"sweep(collect=...): {entry!r} is neither a quantity or signal path of this "
                    f"Problem nor 'cost:<name>'. Paths: {paths}. Cost terms: "
                    f"{[COST_PREFIX + cost for cost in costs] or '(none)'}."
                )
            rows, cols = builder.shape_of(entry)
            if (rows, cols) == (1, 1):
                layout.append((entry, (entry,)))
            else:
                layout.append((entry, tuple(f"{entry}[{i}]" for i in range(rows * cols))))
        self._check_names([column for _, columns in layout for column in columns])
        self._layout = tuple(layout)

    def columns(self) -> tuple:
        """The collected column names; one per entry if no Problem was ever built to check."""
        if self._layout is None:
            return self._entries
        return tuple(column for _, columns in self._layout for column in columns)

    def values(self, problem, result) -> dict:
        """``{column: float}`` at a successful result."""
        evaluated = None
        if any(not entry.startswith(COST_PREFIX) for entry, _ in self._layout):
            evaluated = problem.evaluate(result)
        out = {}
        for entry, columns in self._layout:
            if entry.startswith(COST_PREFIX):
                name = entry[len(COST_PREFIX):]
                if name not in result.cost_terms:
                    raise ValueError(
                        f"sweep(collect=...): the Problem at this point has no cost term "
                        f"{name!r}; its cost terms: {list(result.cost_terms)}."
                    )
                out[entry] = float(result.cost_terms[name])
                continue
            if entry not in evaluated:
                raise ValueError(
                    f"sweep(collect=...): the Problem at this point has no path {entry!r}; the "
                    f"columns were fixed by the first Problem the target returned."
                )
            flat = np.asarray(evaluated[entry], dtype=float).ravel(order="F")
            if flat.size != len(columns):
                raise ValueError(
                    f"sweep(collect=...): {entry!r} has {flat.size} element(s) at this point and "
                    f"{len(columns)} in the first Problem the target returned, which fixed the "
                    f"columns. Collect a quantity whose shape does not change with the axes."
                )
            for column, value in zip(columns, flat):
                out[column] = float(value)
        return out

    def _check_names(self, columns) -> None:
        clashing = [column for column in columns if column in self._reserved]
        if clashing:
            raise ValueError(
                f"sweep(collect=...): {clashing} would be named like an axis or a status column "
                f"({list(self._reserved)}). An axis value is already its own column; drop the "
                f"entry."
            )


# --- values -----------------------------------------------------------------------------------


def _axes(axes) -> tuple:
    """``((name, values), ...)`` in axis order, every value a Python scalar."""
    if isinstance(axes, (set, frozenset)):
        raise TypeError(
            "sweep(axes=...) is a set, which has no order, so the grid order would change from "
            "run to run. Pass a dict {name: values} or a list of (name, values) pairs."
        )
    if isinstance(axes, Mapping):
        pairs = list(axes.items())
    elif isinstance(axes, (list, tuple)):
        pairs = []
        for index, item in enumerate(axes):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise TypeError(
                    f"sweep(axes=[...]): item {index} is {item!r}, not a (name, values) pair."
                )
            pairs.append((item[0], item[1]))
    else:
        raise TypeError(
            f"sweep(axes=...) takes an ordered mapping {{name: values}} or a list of "
            f"(name, values) pairs, got {type(axes).__name__}."
        )
    if not pairs:
        raise ValueError(
            "sweep(axes=...) is empty; give at least one axis, e.g. {'limit': [100.0, 150.0]}."
        )
    out, seen = [], []
    for name, values in pairs:
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"sweep axis names are strings (an instance path, or a keyword of a callable "
                f"target), got {name!r}."
            )
        if name in seen:
            raise ValueError(f"sweep axis {name!r} is given twice; merge its values into one axis.")
        seen.append(name)
        out.append((name, _axis_values(name, values)))
    return tuple(out)


def _axis_values(name, values) -> tuple:
    if isinstance(values, (set, frozenset)):
        raise TypeError(
            f"sweep axis {name!r} is a {type(values).__name__}, which has no order, so the row "
            f"order would change from run to run. Pass a list, tuple, range or 1-D array "
            f"(sorted(...) if a set is what you have)."
        )
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(
                f"sweep axis {name!r} is a {values.ndim}-D array; give a 1-D array, one value per "
                f"grid step."
            )
        items = values.tolist()
    elif isinstance(values, (list, tuple, range)):
        items = list(values)
    else:
        raise TypeError(
            f"sweep axis {name!r} takes an ordered sequence of values (list, tuple, range or 1-D "
            f"array), got {type(values).__name__}."
        )
    if not items:
        raise ValueError(
            f"sweep axis {name!r} has no values, which makes the grid empty. Give it at least one "
            f"value, or drop the axis."
        )
    return tuple(_scalar(value, f"sweep axis {name!r}") for value in items)


def _scalar(value, what: str, *, allow_none=False):
    """A Python bool, int, float or str (numpy scalars converted); ``None`` if allowed."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None and allow_none:
        return None
    if isinstance(value, str):
        if "\r" in value:
            raise ValueError(
                f"{what}: {value!r} holds a carriage return. Python 3.10 and 3.12 quote a bare "
                f"'\\r' differently in CSV, which would break the byte-stable to_csv(); remove it."
            )
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    kinds = "None, a bool, a number or a string" if allow_none else "a number, a bool or a string"
    raise TypeError(f"{what}: {value!r} is not a scalar ({kinds}).")


def _cell(value, column, index) -> str:
    """One CSV cell, spelled the same by every interpreter."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, str):
        if "\r" in value:
            raise ValueError(
                f"SweepTable.to_csv: row {index}, column {column!r} holds a carriage return, "
                f"which Python 3.10 and 3.12 quote differently; remove it from the row."
            )
        return value
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        return repr(float(value))
    raise TypeError(
        f"SweepTable.to_csv: row {index}, column {column!r} holds {value!r}, which is not a "
        f"scalar (None, a bool, a number or a string)."
    )


def _describe(exc) -> str:
    """``"<Type>: <first line>"``; ``splitlines`` also drops any ``\\r``."""
    lines = str(exc).splitlines()
    first = lines[0].strip() if lines else ""
    name = type(exc).__name__
    return f"{name}: {first}" if first else name


def _optional_float(value):
    return None if value is None else float(value)
