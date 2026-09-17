"""
Layer 3 — Agent Type base class and supporting dataclasses.

The two-phase declare/build lifecycle
--------------------------------------
1. declare()  — Agent inspects its config and returns a list of
                QuantityDeclaration objects.  No CasADi symbols are created;
                no solver interaction occurs.  The compiler uses these
                declarations to assign variable vs. parameter roles based on
                YAML overrides (or Python dicts in the stub compiler).

2. build(symbols) — The compiler has already created MX symbols for every
                    declared quantity and passes them here.  The agent builds
                    internal expressions using Layer 2 functions, populates
                    its resolvable namespace, and returns ConstraintDeclaration
                    objects for the compiler to register with the solver.

Agents are fully decoupled from SolverBackend.  They never call
solver.add_variable(), solver.add_constraint(), etc.  All solver interaction
is mediated by the compiler.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import casadi as ca
import numpy as np

from machina.blocks.descriptor import _VALID_SEMANTIC_TYPES, SymbolDescriptor

# ---------------------------------------------------------------------------
# Validation sets
# ---------------------------------------------------------------------------

_VALID_ROLES = frozenset({'flexible', 'always_variable', 'always_parameter'})
_VALID_DEFAULT_ROLES = frozenset({'variable', 'parameter'})


# ---------------------------------------------------------------------------
# QuantityDeclaration
# ---------------------------------------------------------------------------

@dataclass
class QuantityDeclaration:
    """
    Describes a single quantity an agent needs.

    Produced during declare(), consumed by the compiler to create MX symbols
    (as either decision variables or parameters) before calling build().

    Fields
    ------
    path : str
        Path relative to the agent, e.g. 'orbital/sma'.  The compiler
        prepends the agent name to form the full solver-level name:
        '{agent_name}/{path}'.  Must not start with '/'.

    shape : tuple
        Shape of the quantity, e.g. (1, 1) for scalar, (6, 1) for a state
        vector.  Must match the MX symbol the compiler creates.

    semantic_type : str
        One of: 'scalar', 'vector', 'trajectory', 'indexed_set', 'time_grid'.

    default_value : float or np.ndarray
        Used as initial guess when assigned as a decision variable, or as the
        fixed numeric value when assigned as a parameter.  The compiler warns
        whenever this default is used in place of a user-provided override.

    lb, ub : float or np.ndarray
        Lower and upper bounds.  Only meaningful when the quantity is assigned
        the variable role.  Ignored for parameters.

    description : str
        Human-readable description for logging and documentation.

    units : str, optional
        Advisory only.  Not enforced programmatically.

    frame : str, optional
        Reference frame for spatial vectors ('ECI', 'LVLH', 'body').
        None for non-spatial quantities.

    role : str
        Who decides variable vs. parameter assignment:
        - 'flexible'         : YAML/compiler decides; default_role is the fallback.
        - 'always_variable'  : Internal optimization DOF; compiler cannot fix it.
        - 'always_parameter' : Physical constant; compiler cannot free it.

    default_role : str
        Fallback when role='flexible' and the YAML/compiler provides no override.
        Either 'variable' or 'parameter'.  Has no effect when role != 'flexible'.
    """

    path: str
    shape: tuple
    semantic_type: str
    default_value: float | np.ndarray
    lb: float | np.ndarray
    ub: float | np.ndarray
    description: str
    units: str = None
    frame: str = None
    role: str = 'flexible'
    default_role: str = 'variable'

    def __post_init__(self):
        if self.role not in _VALID_ROLES:
            raise ValueError(
                f"QuantityDeclaration '{self.path}': invalid role '{self.role}'. "
                f"Must be one of {sorted(_VALID_ROLES)}."
            )
        if self.default_role not in _VALID_DEFAULT_ROLES:
            raise ValueError(
                f"QuantityDeclaration '{self.path}': invalid default_role "
                f"'{self.default_role}'. Must be one of {sorted(_VALID_DEFAULT_ROLES)}."
            )
        if self.semantic_type not in _VALID_SEMANTIC_TYPES:
            raise ValueError(
                f"QuantityDeclaration '{self.path}': invalid semantic_type "
                f"'{self.semantic_type}'. Must be one of "
                f"{sorted(_VALID_SEMANTIC_TYPES)}."
            )
        if self.path.startswith('/'):
            raise ValueError(
                f"QuantityDeclaration path must not start with '/': '{self.path}'."
            )
        if self.role != 'flexible' and self.default_role != 'variable':
            warnings.warn(
                f"QuantityDeclaration '{self.path}': default_role='{self.default_role}' "
                f"has no effect when role='{self.role}' (not 'flexible').",
                UserWarning,
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# ConstraintDeclaration
# ---------------------------------------------------------------------------

@dataclass
class ConstraintDeclaration:
    """
    Describes a constraint returned by build() for the compiler to register.

    The compiler calls solver.add_constraint(expr, lb, ub, name=...) for each
    declaration, prepending the agent name to form the full constraint name.

    No __post_init__ validation — the compiler validates shapes when it calls
    solver.add_constraint().  Not frozen (ca.MX is not hashable).

    Fields
    ------
    expr : ca.MX
        MX expression for the constraint.  Can be scalar or vector-valued.
        For equality constraints set lb == ub (IPOPT auto-detects).

    lb, ub : float or np.ndarray
        Bounds.  Scalar broadcasts to all rows of expr.
        Use -np.inf / np.inf for one-sided constraints.

    name : str
        Agent-relative name.  The compiler prepends the agent name:
        '{agent_name}/{name}'.

    description : str
        Human-readable description for logging and debugging.
    """

    expr: ca.MX
    lb: float | np.ndarray
    ub: float | np.ndarray
    name: str
    description: str


# ---------------------------------------------------------------------------
# AgentType ABC
# ---------------------------------------------------------------------------

class AgentType(ABC):
    """
    Abstract base class for all Layer 3 agent types.

    Subclasses implement declare() and build() to encode structural domain
    knowledge about their subsystem.

    Parameters
    ----------
    name : str
        Unique agent name.  Used as path prefix for all quantities registered
        with the solver (e.g. 'flyby_sat').
    config : dict
        Agent-specific configuration dict.  Structure is defined by each
        subclass.  The problem compiler merges YAML overrides before
        constructing the agent.
    """

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self._namespace: dict[str, SymbolDescriptor] = {}
        self._built = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def declare(self) -> list[QuantityDeclaration]:
        """
        Declare all quantities this agent needs.

        Reads self.config to determine which quantities exist.  No MX symbols
        are created and no solver interaction occurs.

        Returns
        -------
        list[QuantityDeclaration]
            Every quantity the agent will use during build().
        """
        ...

    @abstractmethod
    def build(self, symbols: dict[str, ca.MX]) -> list[ConstraintDeclaration]:
        """
        Build internal expressions, populate the namespace, return constraints.

        Parameters
        ----------
        symbols : dict[str, ca.MX]
            Maps every declared quantity path to an MX symbol created by the
            compiler (as either a variable or parameter).  Keys are the same
            strings used in QuantityDeclaration.path.

        Returns
        -------
        list[ConstraintDeclaration]
            Structural constraints inherent to this agent's physics.  The
            compiler registers them with the solver.

        Post-conditions
        ---------------
        - self._namespace is populated with all externally resolvable
          quantities (variables, parameters, and computed expressions).
        - self._built is set to True as the final step.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete helpers (provided by base class)
    # ------------------------------------------------------------------

    def resolve(self, path: str) -> SymbolDescriptor:
        """
        Look up a path in this agent's namespace.

        The compiler strips the agent-name prefix before calling this, so
        path is relative to the agent (e.g. 'products/cmi_conus/ttp', not
        'flyby_sat/products/cmi_conus/ttp').

        Parameters
        ----------
        path : str
            Relative path string using '/' as separator.

        Returns
        -------
        SymbolDescriptor

        Raises
        ------
        RuntimeError
            If called before build().
        KeyError
            If path is not in the namespace (error message lists available paths).
        """
        if not self._built:
            raise RuntimeError(
                f"Agent '{self.name}': resolve() called before build(). "
                "Call build(symbols) first."
            )
        if path not in self._namespace:
            available = sorted(self._namespace.keys())
            raise KeyError(
                f"Agent '{self.name}': path '{path}' not found in namespace. "
                f"Available paths: {available}"
            )
        return self._namespace[path]

    def list_paths(self) -> list[str]:
        """
        Return all resolvable paths in sorted order.

        Safe to call before build() — returns an empty list rather than
        raising.  Includes decision variables, parameters, and computed
        expressions once build() has been called.
        """
        return sorted(self._namespace.keys())

    def __repr__(self) -> str:
        state = 'built' if self._built else 'not built'
        n_paths = len(self._namespace)
        return (
            f"{type(self).__name__}(name='{self.name}', {state}, "
            f"{n_paths} namespace entries)"
        )
