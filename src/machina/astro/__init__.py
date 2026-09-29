"""
machina.astro -- orbital mechanics pack.

Modified equinoctial elements, orbit geometry, formation relative motion,
antenna-beam geometry and the coverage component built on them. Importing this
package declares its frames (``eci``, ``ecef``, ``lvlh``) and signals into the
default signal registry and registers its factories:

* the seven ``transform.*`` (:mod:`~machina.astro.transforms`);
* the two ``geometry.*`` of :mod:`~machina.astro.geometry`;
* ``cost.smooth_coverage`` (:mod:`~machina.astro.coverage`);
* ``transform.roe_to_rtn``, ``geometry.rn_min_separation`` and
  ``util.synodic_period`` (:mod:`~machina.astro.relative`: period-matched
  formation about a near-circular, near-equatorial chief, and the synodic
  period of two circular orbits);
* ``geometry.off_axis_angle``, ``cost.smooth_in_cone`` and
  ``cost.mean_beam_duty`` (:mod:`~machina.astro.beam`: antenna-cone geometry).

``import machina`` never imports it, so nothing in the core depends on an orbit
existing.
"""

from machina.astro import coverage, geometry, signals, transforms  # noqa: F401

# isort: split
# Phase 4c modules, appended so the Phase 3 factories keep registering first.
from machina.astro import beam, relative  # noqa: F401
from machina.astro.components.single_sat_coverage import SingleSatCoverage

__all__ = ["signals", "SingleSatCoverage"]
