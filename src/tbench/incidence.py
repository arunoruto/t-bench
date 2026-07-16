"""Deterministic incidence-direction generation for cross-section averaging.

Uses scipy's Halton sequence to produce N well-distributed points on the
unit sphere, converted to (polar_deg, azimuthal_deg) pairs usable by both
MSTM's ``set_incident()`` and FaSTMM2's ``_rotate_positions_for_incident()``.

Both adapters call the same function with the same ``n`` and ``seed`` from
``ClusterRequest`` / ``SweepRequest``, so every incidence-averaged solve
uses exactly the same set of directions.
"""

from __future__ import annotations

import numpy as np
from scipy.stats.qmc import Halton


def generate_incidence_directions(
    n: int,
    seed: int = 0,
) -> list[tuple[float, float]]:
    """Return *n* Halton-sphere directions as ``(polar_deg, azimuthal_deg)``.

    The Halton sequence with bases (2, 3) maps the first coordinate to
    cos(theta) uniformly on [-1, 1] and the second to phi uniformly on
    [0, 2π), producing a low-discrepancy spherical distribution.
    The same seed always reproduces the same sequence.
    """
    if n <= 0:
        return []

    sampler = Halton(d=2, seed=seed)
    points = sampler.random(n)

    polar = np.degrees(np.arccos(1.0 - 2.0 * points[:, 0]))
    azimuthal = np.degrees(2.0 * np.pi * points[:, 1])

    return [(float(p), float(a)) for p, a in zip(polar, azimuthal)]
