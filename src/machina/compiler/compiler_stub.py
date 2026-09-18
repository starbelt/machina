"""
Layer 4 (stub) — Python-driven declare → assign → build → register lifecycle.

This is the Phase 3a compiler stub.  It implements the full agent lifecycle
mediator without YAML parsing.  Callers supply agents and optional override
dicts directly in Python.  The full Layer 4 compiler (Phase 4) will add YAML
parsing, a schema validator, and richer diagnostics on top of this foundation.

Lifecycle
---------
1. compiler.add_agent(agent)           -- register one or more agents
2. compiler.compile(overrides)         -- declare → assign roles → build → register
3. compiler.add_cost(expr, name)       -- wire cost terms (MX expressions from resolve())
4. compiler.build_solver(opts)         -- assemble the CasADi NLP
5. compiler.solve()                    -- call IPOPT, return SolutionResult

Overrides dict format
---------------------
overrides = {
    '{agent_name}/{qty_path}': {
        'role':          'variable' | 'parameter',  # override flexible role
        'value':         float | np.ndarray,         # override default_value
        'lb':            float | np.ndarray,         # override lower bound
        'ub':            float | np.ndarray,         # override upper bound
        'initial_guess': float | np.ndarray,         # override initial guess
        'scale':         float | np.ndarray,         # nominal magnitude (variables)
    },
    '{agent_name}/{constraint_name}': {
        'scale':         float,                      # nominal magnitude of the constraint
    },
    ...
}

Keys that are absent from the override dict for a given quantity fall back to
the agent's declared defaults.  A UserWarning is emitted for each quantity
where no override was provided (to help users discover quantities they may
have forgotten to tune).
"""

import warnings

import casadi as ca
import numpy as np

from machina.agents.agent_type import AgentType, QuantityDeclaration
from machina.blocks.descriptor import SymbolDescriptor
from machina.solver.backend import SolverBackend


class CompilerStub:
    """
    Minimal Python-driven compiler that mediates the agent lifecycle.

    Parameters
    ----------
    solver : SolverBackend, optional
        Pre-existing backend instance.  If None, a fresh one is created.
    solver_opts : dict, optional
        Options forwarded to SolverBackend.__init__ when solver is None.
        Ignored if solver is provided.
    """

    def __init__(self, solver: SolverBackend = None, solver_opts: dict = None):
        if solver is not None:
            self._solver = solver
        else:
            self._solver = SolverBackend(solver_opts=solver_opts or {})

        self._agents: list[AgentType] = []
        self._compiled = False

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def add_agent(self, agent: AgentType) -> None:
        """
        Register an agent.  Must be called before compile().

        Raises
        ------
        RuntimeError
            If compile() has already been called.
        ValueError
            If another agent with the same name is already registered.
        """
        if self._compiled:
            raise RuntimeError(
                "Cannot add agents after compile() has been called."
            )
        existing_names = {a.name for a in self._agents}
        if agent.name in existing_names:
            raise ValueError(
                f"An agent named '{agent.name}' is already registered."
            )
        self._agents.append(agent)

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def compile(self, overrides: dict = None) -> None:
        """
        Run the declare → assign → build → register lifecycle for all agents.

        For each agent, in registration order:
          1. Calls agent.declare() to get QuantityDeclarations.
          2. Resolves the role and effective bounds/values for each quantity.
          3. Creates MX symbols via solver.add_variable() or add_parameter().
          4. Calls agent.build(symbols) to get ConstraintDeclarations.
          5. Registers each constraint with the solver backend.

        Parameters
        ----------
        overrides : dict, optional
            Per-quantity override dicts.  Keys are full paths
            '{agent_name}/{qty_path}'.  See module docstring for the
            per-quantity override key options.

        Raises
        ------
        RuntimeError
            If called more than once.
        """
        if self._compiled:
            raise RuntimeError("compile() has already been called.")

        overrides = overrides or {}

        for agent in self._agents:
            declarations = agent.declare()
            symbols: dict[str, ca.MX] = {}

            for decl in declarations:
                full_path = f"{agent.name}/{decl.path}"
                override = overrides.get(full_path, {})

                role = self._resolve_role(decl, override)
                value = override.get('value', decl.default_value)
                lb = override.get('lb', decl.lb)
                ub = override.get('ub', decl.ub)
                guess = override.get('initial_guess', value)

                # Warn whenever the agent's declared default is used —
                # the user may want to tune these explicitly.
                if not override:
                    warnings.warn(
                        f"CompilerStub: '{full_path}' using default value "
                        f"{decl.default_value!r} [{decl.units or 'no units'}]. "
                        "Provide an override to suppress this warning.",
                        UserWarning,
                        stacklevel=2,
                    )

                if role == 'variable':
                    sym = self._solver.add_variable(
                        full_path, decl.shape,
                        lb=lb, ub=ub, initial_guess=guess,
                        scale=override.get('scale', 1.0),
                    )
                else:  # 'parameter'
                    # The backend stores the value and flattens matrix
                    # values column-major to match ca.vec.
                    sym = self._solver.add_parameter(
                        full_path, decl.shape, value=value,
                    )

                symbols[decl.path] = sym

            constraints = agent.build(symbols)

            for cdecl in constraints:
                cname = f"{agent.name}/{cdecl.name}"
                self._solver.add_constraint(
                    cdecl.expr,
                    lb=cdecl.lb,
                    ub=cdecl.ub,
                    name=cname,
                    scale=overrides.get(cname, {}).get('scale', 1.0),
                )

        self._compiled = True

    # ------------------------------------------------------------------
    # Post-compile wiring (costs + cross-agent constraints)
    # ------------------------------------------------------------------

    def add_cost(self, expr: ca.MX, name: str = None) -> None:
        """
        Register a cost term with the solver.

        The caller is responsible for resolving path references through
        agents (via self.resolve()) before constructing the expression.

        Parameters
        ----------
        expr : ca.MX
            Scalar MX expression.
        name : str, optional
            Label for debugging and the visualization tool.

        Raises
        ------
        RuntimeError
            If compile() has not been called yet.
        """
        if not self._compiled:
            raise RuntimeError(
                "Call compile() before add_cost()."
            )
        self._solver.add_cost(expr, name=name)

    def add_constraint(
        self,
        expr: ca.MX,
        lb: float = -np.inf,
        ub: float = np.inf,
        name: str = None,
    ) -> None:
        """
        Register a cross-agent or problem-level constraint directly.

        Parameters
        ----------
        expr : ca.MX
            MX expression for the constraint (scalar or vector).
        lb, ub : float
            Bounds.
        name : str, optional
            Label for debugging.

        Raises
        ------
        RuntimeError
            If compile() has not been called yet.
        """
        if not self._compiled:
            raise RuntimeError(
                "Call compile() before add_constraint()."
            )
        self._solver.add_constraint(expr, lb=lb, ub=ub, name=name)

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def resolve(self, agent_name: str, path: str) -> SymbolDescriptor:
        """
        Look up a namespace path in the named agent.

        Parameters
        ----------
        agent_name : str
            Name used when the agent was registered.
        path : str
            Agent-relative path (e.g. 'orbital/sma').

        Returns
        -------
        SymbolDescriptor

        Raises
        ------
        KeyError
            If no agent with agent_name exists, or if the path is not found.
        """
        for agent in self._agents:
            if agent.name == agent_name:
                return agent.resolve(path)
        raise KeyError(
            f"CompilerStub: no agent named '{agent_name}'. "
            f"Registered agents: {[a.name for a in self._agents]}"
        )

    # ------------------------------------------------------------------
    # Build and solve
    # ------------------------------------------------------------------

    def build_solver(self, opts: dict = None) -> None:
        """
        Assemble the CasADi NLP and create the solver object.

        Must be called after compile() and after all add_cost() /
        add_constraint() calls.

        Parameters
        ----------
        opts : dict, optional
            Solver option overrides (e.g. {'ipopt.print_level': 0}).

        Raises
        ------
        RuntimeError
            If compile() has not been called yet.
        """
        if not self._compiled:
            raise RuntimeError(
                "Call compile() before build_solver()."
            )
        self._solver.build(opts=opts)

    def solve(self):
        """
        Invoke the solver and return a SolutionResult.

        Parameter values were stored on the backend at compile() time, so no
        parameter vector needs to be assembled here.

        Returns
        -------
        SolutionResult

        Raises
        ------
        RuntimeError
            If build_solver() has not been called yet.
        """
        if not self._solver.is_built:
            raise RuntimeError(
                "Call build_solver() before solve()."
            )
        return self._solver.solve()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def solver(self) -> SolverBackend:
        """Direct access to the underlying SolverBackend (for advanced use)."""
        return self._solver

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_role(self, decl: QuantityDeclaration, override: dict) -> str:
        """Return the effective role ('variable' or 'parameter') for a quantity."""
        if decl.role == 'always_variable':
            return 'variable'
        if decl.role == 'always_parameter':
            return 'parameter'
        # flexible — check override, then fall back to default_role
        if 'role' in override:
            r = override['role']
            if r not in ('variable', 'parameter'):
                raise ValueError(
                    f"Override 'role' must be 'variable' or 'parameter', got '{r}'."
                )
            return r
        return decl.default_role

    def __repr__(self) -> str:
        n = len(self._agents)
        state = 'compiled' if self._compiled else 'not compiled'
        return f"CompilerStub({n} agent(s), {state})"
