"""Wavelength sweeps: one cluster geometry + one material, evaluated
across a range of wavelengths -- expands into a list of ClusterRequest
(schema.py's per-wavelength request) for the runner to solve one at a
time.

Wavelengths are always in micrometers here, matching refidxdb's own
`interpolate()` convention (confirmed directly against real cached
refractiveindex.info data -- SiO2 at target=[0.5, 1.0] gives n~1.468/
1.459, correct for fused silica in that range) -- so cluster coords/
radii must also be in micrometers for the resulting wavenumber
(k = 2*pi/wavelength_um) to be dimensionally consistent with them.

Both sides are bare numbers meant as micrometers, not SI meters -- a
wavelengths_um entry of 0.5 means 0.5 um, not 0.5 m, and coords/radii
need the matching micrometers scale (see geometry.py's load_positions
docstring for common source-unit conversions, e.g. nm -> 1e-3, not the
SI nm -> meters factor of 1e-9). Getting this wrong on only one side
(e.g. scaling radii to meters while wavelengths stay in micrometers)
silently produces a wavenumber * radius size parameter that's off by
whatever power of ten was missed -- if it lands far enough below 1,
MSTM crashes the whole process rather than raising a normal error (see
adapters/mstm_python.py's MIN_SIZE_PARAMETER guard).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, model_validator

from tbench.schema import ClusterRequest


class MaterialSpec(BaseModel):
    """A single material, either a fixed (dispersion-free) refractive
    index or a refidxdb-backed dispersive lookup. Applied uniformly to
    every sphere in a sweep (a benchmark cluster is assumed to be one
    material) -- per-sphere materials aren't supported by SweepRequest;
    build a list[ClusterRequest] by hand for that.

    Exactly one of the following must be given:
      - refractive_index: a fixed (n, k)
      - refidxdb_source + refidxdb_catalog_path: a refidxdb database name
        ("refidx" or "aria") plus a cache-relative path -- normally taken
        straight from that database's own catalog() (see
        refidxdb.DATABASES[source].catalog()), which is what the
        dashboard's material picker uses so users choose from a real,
        browsable list instead of typing a URL freehand.
      - refidxdb_url: a refractiveindex.info or eodg.atm.ox.ac.uk (ARIA)
        URL, routed through refidxdb.Handler -- kept for scripts that
        already have a URL on hand (e.g. copied from the website).
      - refidxdb_path: a local .csv or .dat file in refidxdb's own
        generic format (not a raw refractiveindex.info .yml -- those
        need refidxdb_source+refidxdb_catalog_path or refidxdb_url
        instead, same restriction refidxdb.Handler itself has).
    """

    refractive_index: tuple[float, float] | None = None
    refidxdb_source: Literal["refidx", "aria"] | None = None
    refidxdb_catalog_path: str | None = None
    refidxdb_url: str | None = None
    refidxdb_path: str | None = None

    @model_validator(mode="after")
    def _check_one_source(self) -> MaterialSpec:
        if (self.refidxdb_source is None) != (self.refidxdb_catalog_path is None):
            raise ValueError(
                "refidxdb_source and refidxdb_catalog_path must be given together"
            )
        catalog_source = self.refidxdb_source is not None
        sources = (self.refractive_index, catalog_source or None, self.refidxdb_url, self.refidxdb_path)
        if sum(s is not None for s in sources) != 1:
            raise ValueError(
                "Specify exactly one of refractive_index, "
                "(refidxdb_source + refidxdb_catalog_path), refidxdb_url, refidxdb_path"
            )
        return self

    def refractive_index_at(
        self, wavelengths_um: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.complex128]:
        """Complex refractive index (n + ik), one value per wavelength."""
        wavelengths_um = np.asarray(wavelengths_um, dtype=np.float64)
        if self.refractive_index is not None:
            n, k = self.refractive_index
            return np.full(wavelengths_um.shape, complex(n, k))

        if self.refidxdb_source is not None:
            from refidxdb import DATABASES

            source = DATABASES[self.refidxdb_source](path=self.refidxdb_catalog_path)
            return source.interpolate(target=wavelengths_um, as_complex=True)

        from refidxdb import Handler

        handler = (
            Handler(url=self.refidxdb_url)
            if self.refidxdb_url is not None
            else Handler(path=self.refidxdb_path)
        )
        return handler.interpolate(target=wavelengths_um, as_complex=True)


class SweepRequest(BaseModel):
    """Cluster geometry (fixed across the sweep) + material + wavelength
    range + shared solver settings. expand_sweep() turns this into one
    ClusterRequest per wavelength, all with the same geometry."""

    coords: list[tuple[float, float, float]]
    radii: list[float]
    material: MaterialSpec
    wavelengths_um: list[float]

    medium_refractive_index: tuple[float, float] = (1.0, 0.0)
    incident_polar_deg: float = 0.0
    incident_azimuthal_deg: float = 0.0
    n_theta: int = 181
    n_phi: int = 1
    tolerance: float = 1e-4
    max_iterations: int = 2000
    mstm_mie_eps: float = 1e-10
    mstm_translation_eps: float = 1e-8
    formulation: int = 0
    mlfmm_accuracy: int = 2
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_lengths(self) -> SweepRequest:
        if len(self.coords) != len(self.radii):
            raise ValueError(
                f"coords ({len(self.coords)}) and radii ({len(self.radii)}) "
                "must have the same length"
            )
        if not self.wavelengths_um:
            raise ValueError("wavelengths_um must not be empty")
        return self

    @property
    def n_spheres(self) -> int:
        return len(self.radii)


def expand_sweep(sweep: SweepRequest) -> list[ClusterRequest]:
    """One ClusterRequest per wavelength in the sweep, same geometry
    throughout, refractive index taken from the material at that
    wavelength and broadcast to every sphere."""
    n = sweep.n_spheres
    nk = sweep.material.refractive_index_at(np.asarray(sweep.wavelengths_um))

    requests = []
    for wl_um, m in zip(sweep.wavelengths_um, nk):
        requests.append(
            ClusterRequest(
                coords=sweep.coords,
                radii=sweep.radii,
                refractive_index=[(float(m.real), float(m.imag))] * n,
                wavenumber=2.0 * np.pi / wl_um,
                medium_refractive_index=sweep.medium_refractive_index,
                incident_polar_deg=sweep.incident_polar_deg,
                incident_azimuthal_deg=sweep.incident_azimuthal_deg,
                n_theta=sweep.n_theta,
                n_phi=sweep.n_phi,
                tolerance=sweep.tolerance,
                max_iterations=sweep.max_iterations,
                mstm_mie_eps=sweep.mstm_mie_eps,
                mstm_translation_eps=sweep.mstm_translation_eps,
                formulation=sweep.formulation,
                mlfmm_accuracy=sweep.mlfmm_accuracy,
                extra=sweep.extra,
            )
        )
    return requests
