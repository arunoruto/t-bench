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
    """MSTM-only: per-sphere Mie coefficient truncation control.

    Positive value = adaptive convergence tolerance (e.g. 1e-10 is tighter
    than pymstm's own library default of 1e-6 -- confirmed needed for
    touching/near-touching spheres deep in the Rayleigh regime, where
    looser defaults under-truncate near-field coupling badly enough to
    flip Csca's sign relative to FaSTMM2's independent computation).

    Negative value = fixed number of Mie terms per sphere (e.g. -8 locks
    truncation to exactly 8 terms, matching YASF-new's approach). A fixed
    lmax prevents the adaptive convergence from creating inconsistencies
    between the Mie coefficient truncation and the T-matrix expansion
    order -- the most reliable way to avoid non-physical negative Csca.
    The adapters cast negative values to int before passing to MSTM.

    See mstm_translation_eps."""
    mstm_translation_eps: float = 1e-8
    """MSTM-only: translation-addition-theorem convergence tolerance
    (controls near-field coupling accuracy between spheres, as distinct
    from mstm_mie_eps's per-sphere truncation). Tighter than pymstm's own
    library default (1e-5) for the same touching-sphere reason as
    mstm_mie_eps -- tightening both moved a real test case's Cext
    discrepancy against FaSTMM2 from ~25% to ~12% and fixed Csca's sign;
    going tighter still (1e-12/1e-10) changed nothing further, i.e. this
    already converges. Ignored by FaSTMM2."""

    formulation: int = 0
    """FaSTMM2-only: 0=STMM, 1=FaSTMM, 2=FaSTMM2 (MLFMM). Ignored by MSTM.
    Default is STMM (exact, no octree/multipole acceleration) rather than
    MLFMM -- confirmed on a real two-sphere case (k=12.57, non-touching)
    that MLFMM alone introduces a genuine ~2% error vs MSTM that vanishes
    completely with STMM (0.0000%). For touching/near-touching clusters
    this doesn't matter (formulation made no measurable difference on a
    128-particle touching aggregate -- that disagreement comes from
    near-field coupling, not the MLFMM approximation). MLFMM's own
    purpose is speed on very large clusters, but even that didn't hold on
    a 128-particle case (STMM: 2.4s, MLFMM: 6.7s) -- only switch to 2 if
    profiling shows MLFMM actually winning for your cluster size."""
    mlfmm_accuracy: int = 2
    """FaSTMM2-only: MLFMM accuracy, significant digits. Ignored by MSTM."""

    n_incidence_angles: int = 0
    """MSTM and FaSTMM2: number of incidence directions to average over.
    0 = single fixed-orientation solve (default). N > 0 = solve at N
    Halton-sphere directions and return the arithmetic mean of the
    cross sections (Cext, Cabs, Csca). Smooths out per-solve numerical
    noise in MSTM's ``Q_sca = Q_ext + Q_inc - Q_abs`` subtraction that
    can produce non-physical negative Csca for highly absorbing
    clusters (e.g. iron nanoparticles). YASF-new routinely uses 4--40
    incidence directions for the same reason."""
    incidence_seed: int = 0
    """Seed for the Halton sequence when n_incidence_angles > 0. Same
    seed + same n guarantees both tools use identical angle sets."""

    compute_mueller: bool = False
    """MSTM and FaSTMM2: also compute S11 (phase function) and S12 (for
    DoLP = -S12/S11) as a function of scattering angle, stored in
    ``ScatterResult.mueller`` as ``[[theta_deg, S11, S12], ...]`` on a
    uniform 0-180deg grid with ``n_theta`` points (both tools use
    identical angles -- no interpolation needed for comparison; see
    adapters/mstm_python.py's comment for why the per-angle
    ``get_scattering_angle()`` API is used for MSTM instead of its
    faster but unreliable batch ``get_scattering_matrix()``). This is
    cheap post-processing on the already-solved field (not a re-solve),
    but off by default to keep it opt-in."""

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
    """Set only when the originating ClusterRequest had compute_mueller=True.
    A reconciled, tool-independent ``[[theta_deg, S11, S12], ...]`` on a
    uniform 0-180deg grid with n_theta points -- both tools evaluated at
    identical angles, so this compares directly with no interpolation.
    S11 is the phase function, normalized to the standard radiative-
    transfer convention ``integral(S11 dOmega) == 4*pi`` (both tools'
    *raw* S11 instead follow the Bohren-Huffman ``dCsca/dOmega = S11/k^2``
    convention -- confirmed empirically by integrating raw S11 over the
    sphere for a single symmetric sphere and comparing to k^2*Csca -- so
    every adapter rescales by ``4*pi/(k^2*Csca)`` before returning it
    here); DoLP = -S12/S11 (scale-invariant, so this rescaling doesn't
    change it, but S12 is rescaled by the same factor for consistency).
    Not the tools' own native full 4x4 Mueller matrices (those have
    incompatible column/row conventions between MSTM and FaSTMM2 -- see
    adapters/mstm_python.py's and fastmm2_python.py's comments); only
    S11/S12 are extracted since that's what's needed for phase
    function/DoLP comparison."""

    raw: dict[str, Any] = Field(default_factory=dict)
    """Tool-native output, for debugging -- not part of the stable schema."""
