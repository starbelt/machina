class SolverError(RuntimeError):
    """
    Raised when the solver layer cannot honour a request as configured.

    Examples: a discrete variable registered with a plugin that has no integer
    support, an option carrying another plugin's prefix, or an ``nlpsol``
    plugin that is not compiled into the running CasADi build. The message
    always names what to change.
    """
