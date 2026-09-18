"""
Per-plugin knowledge for ``SolverBackend``: default options, quiet options,
warm-start options and integer support.

``nlpsol`` plugins do not share an option vocabulary. IPOPT options live under
``ipopt.``, bonmin's under ``bonmin.``, and handing one plugin another's
prefix is a hard CasADi error ("Unknown option: ipopt"). Everything that is
plugin-specific is kept here so that ``backend.py`` stays plugin-agnostic.

Option precedence, lowest to highest::

    spec.defaults  ->  spec.quiet (verbose=False)  ->  spec.warm_start (warm solver only)
                   ->  SolverBackend(solver_opts=...)  ->  build(opts=...)
"""

from dataclasses import dataclass, field

import casadi as ca

from .errors import SolverError


@dataclass(frozen=True)
class PluginSpec:
    name: str
    defaults: dict = field(default_factory=dict)
    quiet: dict = field(default_factory=dict)
    # Options that make duals passed at call time effective. They are fixed at
    # nlpsol construction, which is why the backend keeps a second solver.
    warm_start: dict = field(default_factory=dict)
    supports_discrete: bool = False

    @property
    def prefix(self) -> str:
        return self.name


_WARM_PUSH = 1e-9

_QRQP_QUIET = {'print_iter': False, 'print_header': False, 'print_info': False}

_SPECS: dict[str, PluginSpec] = {
    'ipopt': PluginSpec(
        name='ipopt',
        # Unchanged from the April 2026 backend so existing problems behave
        # identically.
        defaults={
            'print_time':          True,
            'ipopt.tol':           1e-8,
            'ipopt.max_iter':      2000,
            'ipopt.linear_solver': 'mumps',
            'ipopt.mu_strategy':   'adaptive',
            'ipopt.print_level':   5,
        },
        quiet={
            'print_time':        False,
            'ipopt.print_level': 0,
            'ipopt.sb':          'yes',
        },
        warm_start={
            'ipopt.warm_start_init_point':       'yes',
            'ipopt.warm_start_bound_push':       _WARM_PUSH,
            'ipopt.warm_start_bound_frac':       _WARM_PUSH,
            'ipopt.warm_start_slack_bound_push': _WARM_PUSH,
            'ipopt.warm_start_slack_bound_frac': _WARM_PUSH,
            'ipopt.warm_start_mult_bound_push':  _WARM_PUSH,
            'ipopt.mu_init':                     1e-6,
        },
    ),
    'bonmin': PluginSpec(
        name='bonmin',
        # bonmin's NLP0012I/NLP0014I lines cannot be silenced through options
        # in CasADi 3.7; the Cbc and branch-and-bound logs can.
        quiet={
            'print_time':            False,
            'bonmin.bb_log_level':   0,
            'bonmin.nlp_log_level':  0,
            'bonmin.lp_log_level':   0,
            'bonmin.milp_log_level': 0,
            'bonmin.oa_log_level':   0,
            'bonmin.fp_log_level':   0,
        },
        supports_discrete=True,
    ),
    'sqpmethod': PluginSpec(
        name='sqpmethod',
        # qrqp ships inside CasADi and can be silenced; the vendor default
        # (qpoases) prints a licence banner on every construction.
        defaults={'qpsol': 'qrqp', 'qpsol_options': {'error_on_fail': False}},
        quiet={
            'print_time':      False,
            'print_header':    False,
            'print_iteration': False,
            'print_status':    False,
            'qpsol_options':   dict(_QRQP_QUIET),
        },
    ),
    'fatrop': PluginSpec(
        name='fatrop',
        quiet={'print_time': False, 'fatrop.print_level': 0},
    ),
    'knitro': PluginSpec(name='knitro', supports_discrete=True),
    'snopt': PluginSpec(name='snopt'),
}


def spec_for(name: str) -> PluginSpec:
    """Return the spec for a plugin; an unknown plugin gets an empty one."""
    if name in _SPECS:
        return _SPECS[name]
    return PluginSpec(name=name)


def known_plugins() -> list[str]:
    return list(_SPECS)


def require_available(name: str) -> None:
    """Raise ``SolverError`` if the plugin is not compiled into this CasADi."""
    if ca.has_nlpsol(name):
        return
    raise SolverError(
        f"nlpsol plugin '{name}' is not available in this CasADi build "
        f"({ca.__version__}). Pass one of the available plugins to "
        f"SolverBackend(solver=...), e.g. 'ipopt'."
    )


def _normalise(opts: dict | None, spec: PluginSpec) -> dict:
    """
    Copy ``opts`` and flatten a nested ``{prefix: {...}}`` block for this
    plugin into dot notation, so nested and flat spellings merge predictably.
    """
    out = {}
    for key, value in (opts or {}).items():
        if key == spec.prefix and isinstance(value, dict):
            for sub, sub_value in value.items():
                out[f'{key}.{sub}'] = sub_value
        else:
            out[key] = value
    return out


def check_foreign_prefix(opts: dict | None, spec: PluginSpec, where: str) -> None:
    """Reject options that belong to a different known plugin."""
    for key in (opts or {}):
        head = key.split('.', 1)[0]
        if head in _SPECS and head != spec.prefix:
            raise SolverError(
                f"Option '{key}' (from {where}) belongs to the '{head}' plugin, "
                f"but this backend uses '{spec.name}'. Remove it, or use the "
                f"'{spec.prefix}.' equivalent."
            )


def strip_prefix(opts: dict | None, prefix: str) -> dict:
    """Drop every option that belongs to ``prefix`` (used when switching plugin)."""
    return {
        key: value for key, value in (opts or {}).items()
        if key.split('.', 1)[0] != prefix
    }


def _merge_into(base: dict, layer: dict) -> None:
    """Merge ``layer`` onto ``base``; dict-valued options merge one level deep."""
    for key, value in layer.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value


def merge_options(spec: PluginSpec, *, verbose: bool, solver_opts: dict | None,
                  build_opts: dict | None, warm: bool = False) -> dict:
    """Assemble the option dict handed to ``ca.nlpsol`` (see module docstring)."""
    check_foreign_prefix(solver_opts, spec, 'solver_opts')
    check_foreign_prefix(build_opts, spec, 'build(opts)')

    user = {}
    _merge_into(user, _normalise(solver_opts, spec))
    _merge_into(user, _normalise(build_opts, spec))

    merged: dict = {}
    _merge_into(merged, {k: (dict(v) if isinstance(v, dict) else v)
                         for k, v in spec.defaults.items()})
    if not verbose:
        _merge_into(merged, {k: (dict(v) if isinstance(v, dict) else v)
                             for k, v in spec.quiet.items()})
    if warm:
        _merge_into(merged, spec.warm_start)

    # The sqpmethod defaults carry options for the qrqp QP solver. If the
    # caller picks another QP solver they do not apply.
    if spec.name == 'sqpmethod' and user.get('qpsol', 'qrqp') != 'qrqp':
        merged.pop('qpsol_options', None)

    _merge_into(merged, user)
    return merged
