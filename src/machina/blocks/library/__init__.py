# Import all factory modules to trigger @register decorators.
# Order matters only if factories depend on each other at import time
# (currently none do, but cost imports before constraint for consistency).
from . import (
    cost,  # noqa: F401
    geometry,  # noqa: F401
    transforms,  # noqa: F401
    util,  # noqa: F401
)
