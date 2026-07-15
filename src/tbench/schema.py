"""
Common request/result schema for comparing MSTM and FaSTMM2.

This is the only thing pymstm/pyfastmm code never sees: neither package
imports or knows about t-bench. t-bench imports *them* (see
adapters/), and the adapters translate between this common schema and
each tool's own native API -- see adapters/base.py for why the shared
interface lives here as adapters rather than as a base class either
tool's own solver class would need to inherit from.

Units: coords/radii are in whatever consistent physical length unit you
like (e.g. micrometers); wavenumber must be in the reciprocal of that
same unit (k = 2*pi/wavelength). Both adapters convert internally to
each tool's own native convention -- MSTM wants pre-scaled size
parameters (x = k*r) instead of a separate wavenumber, FaSTMM2 wants
permittivity (eps = m**2) instead of refractive index directly.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ClusterRequest(BaseModel):
    """A cluster of spherical monomers plus the incident field and solver
    settings needed to compute its fixed-orientation scattering response.
    """

    coords: list[tuple[float, float, float]]
    radii: list[float]
    refractive_index: list[tuple[float, float]]
    """Per-sphere complex refractive index (n_re, n_im)."""

    wavenumber: float
    """k = 2*pi/wavelength, in the same length unit as coords/radii."""

    medium_refractive_index: tuple[float, float] = (1.0, 0.0)
    """Surrounding medium. Honored by MSTM; FaSTMM2 has no background-
    medium model (always vacuum) -- Fastmm2*Adapter warns and ignores a
    non-trivial value here, same as pyfastmm._config's own handling."""

    incident_polar_deg: float = 0.0
    incident_azimuthal_deg: float = 0.0

    n_theta: int = 181
    n_phi: int = 1
    """FaSTMM2-only: angular resolution of the returned Mueller matrix.
    MSTM's own scattering-matrix resolution is set separately (not part
    of this v1 schema, which focuses on cross sections -- see
    ScatterResult's `mueller` field docstring)."""

    tolerance: float = 1e-4
    max_iterations: int = 2000

    mstm_mie_eps: float = 1e-10
    """MSTM-only: per-sphere Mie coefficient convergence tolerance.
    Tighter than pymstm's own library default (1e-6) -- confirmed needed
    for touching/near-touching spheres deep in the Rayleigh regime (e.g.
    a fractal aggregate at nanometer particle scale), where the default
    under-truncates near-field coupling badly enough to flip Csca's sign
    relative to FaSTMM2's independent computation. See mstm_translation_eps."""
    mstm_translation_eps: float = 1e-8
    """MSTM-only: translation-addition-theorem convergence tolerance
    (controls near-field coupling accuracy between spheres, as distinct
    from mstm_mie_eps's per-sphere truncation). Tighter than pymstm's own
    library default (1e-5) for the same touching-sphere reason as
    mstm_mie_eps -- tightening both moved a real test case's Cext
    discrepancy against FaSTMM2 from ~25% to ~12% and fixed Csca's sign;
    going tighter still (1e-12/1e-10) changed nothing further, i.e. this
    already converges. Ignored by FaSTMM2."""

    formulation: int = 2
    """FaSTMM2-only: 0=STMM, 1=FaSTMM, 2=FaSTMM2 (MLFMM). Ignored by MSTM."""
    mlfmm_accuracy: int = 2
    """FaSTMM2-only: MLFMM accuracy, significant digits. Ignored by MSTM."""

    extra: dict[str, Any] = Field(default_factory=dict)
    """Escape hatch for adapter-specific overrides not worth promoting to
    a first-class field yet."""

    @model_validator(mode="after")
    def _check_lengths(self) -> ClusterRequest:
        n = len(self.radii)
        if len(self.coords) != n or len(self.refractive_index) != n:
            raise ValueError(
                f"coords ({len(self.coords)}), radii ({n}), and "
                f"refractive_index ({len(self.refractive_index)}) must all "
                "have the same length"
            )
        return self

    @property
    def n_spheres(self) -> int:
        return len(self.radii)


class ScatterResult(BaseModel):
    """Result of solving one ClusterRequest with one adapter."""

    tool: Literal["mstm", "fastmm2"]
    backend: Literal["python", "cli"]
    adapter_name: str

    c_ext: float
    c_abs: float
    c_sca: float
    asymmetry: float | None = None

    wall_time_seconds: float
    iterations: int | None = None
    n_spheres: int

    mueller: list[list[float]] | None = None
    """Tool-native angular Mueller matrix, if computed -- shape and column
    layout differ between tools (MSTM: (n_angles, 16); FaSTMM2: (N_theta *
    N_phi, 18) with phi/theta as the first two columns), deliberately not
    reconciled into one common grid in this v1 schema. Reserved for a
    future comparison pass; the primary output for now is cross sections
    and timing, which compare directly."""

    raw: dict[str, Any] = Field(default_factory=dict)
    """Tool-native output, for debugging -- not part of the stable schema."""
