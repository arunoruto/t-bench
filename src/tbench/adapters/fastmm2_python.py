"""Fastmm2PythonAdapter -- drives pyfastmm.FaSTMM2 directly, in-process."""

from __future__ import annotations

import time
import warnings

import numpy as np

from tbench.adapters.base import ScattererAdapter
from tbench.incidence import generate_incidence_directions
from tbench.schema import ClusterRequest, ScatterResult


def _rotate_positions_for_incident(
    coords: list[tuple[float, float, float]],
    polar_deg: float,
    azimuthal_deg: float,
) -> list[tuple[float, float, float]]:
    """Rotate cluster coordinates so that solving with FaSTMM2's fixed +z
    incident beam is physically equivalent to illuminating the *unrotated*
    cluster from (polar_deg, azimuthal_deg).

    FaSTMM2's solver always illuminates along +z (x-polarized) with no
    angle control. MSTM's own incident field (mstm-input-37.f90) builds
    the tilted plane wave directly as k_hat = Rz(alpha) . Ry(beta) . z_hat
    -- beta (polar) applied about y first, then alpha (azimuthal) about z.
    To make FaSTMM2's fixed +z beam see the same physical wave relative
    to the cluster, apply the *inverse* of that composition to the
    geometry: R = Ry(-beta) . Rz(-alpha) (Rz innermost, then Ry) -- NOT
    Rz(-beta) . Ry(-alpha). Confirmed empirically: the wrong order agrees
    with MSTM to <0.001% whenever only one of polar/azimuthal is nonzero
    (a single rotation is its own trivial composition, so order doesn't
    matter), but is off by up to ~1.3% once both are simultaneously
    nonzero (order matters once neither factor is the identity) -- this
    order matches MSTM to <0.0002% in that combined case too.
    """
    if polar_deg == 0.0 and azimuthal_deg == 0.0:
        return coords

    theta = np.radians(polar_deg)
    phi = np.radians(azimuthal_deg)

    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    Ry = np.array([[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]])
    cos_p, sin_p = np.cos(-phi), np.sin(-phi)
    Rz = np.array([[cos_p, -sin_p, 0], [sin_p, cos_p, 0], [0, 0, 1]])
    R = Ry @ Rz

    arr = np.asarray(coords)
    rotated = arr @ R.T
    return [tuple(p) for p in rotated]  # type: ignore[return-type]


class Fastmm2PythonAdapter(ScattererAdapter):
    name = "fastmm2-python"

    def is_available(self) -> bool:
        try:
            import pyfastmm  # noqa: F401
        except ImportError:
            return False
        return True

    def solve(self, request: ClusterRequest) -> ScatterResult:
        from pyfastmm import FaSTMM2

        if request.medium_refractive_index != (1.0, 0.0):
            warnings.warn(
                "FaSTMM2 has no background-medium model (always vacuum); "
                f"medium_refractive_index={request.medium_refractive_index} is ignored.",
                stacklevel=2,
            )

        directions = generate_incidence_directions(
            request.n_incidence_angles,
            request.incidence_seed,
        )
        angles_to_solve = (
            directions
            if directions
            else [(request.incident_polar_deg, request.incident_azimuthal_deg)]
        )

        eps = [complex(*ri) ** 2 for ri in request.refractive_index]

        t0 = time.perf_counter()
        f = FaSTMM2()

        c_ext_vals: list[float] = []
        c_abs_vals: list[float] = []
        c_sca_vals: list[float] = []
        asymmetry_vals: list[float] = []
        raw_angles: list[tuple[float, float]] = []
        mueller: list[list[float]] | None = None

        for i, (polar_deg, azimuthal_deg) in enumerate(angles_to_solve):
            raw_angles.append((polar_deg, azimuthal_deg))
            coords = _rotate_positions_for_incident(
                request.coords,
                polar_deg,
                azimuthal_deg,
            )
            result = f.solve(
                coords,
                request.radii,
                eps,
                request.wavenumber,
                N_theta=request.n_theta,
                N_phi=max(1, request.n_phi),
                formulation=request.formulation,
                acc=request.mlfmm_accuracy,
                tol=request.tolerance,
                restart=5,
                max_iter=request.max_iterations,
            )
            c_ext_vals.append(result["c_ext"])
            c_abs_vals.append(result["c_abs"])
            c_sca_vals.append(result["c_ext_minus_c_abs"])
            asymmetry_vals.append(result["asymmetry"])

            # S11(theta)/DoLP don't have a clean meaning averaged across
            # different incidence directions (theta is relative to the
            # incident direction) -- only extract from the first solved
            # orientation, same convention as mstm_python.py.
            if request.compute_mueller and i == 0:
                # columns are [phi, theta(RADIANS -- not degrees despite
                # the theta_deg-sounding convention elsewhere), S11, S12,
                # ...], row-major, phi outermost: the first n_theta rows
                # are exactly the phi=0 cut regardless of N_phi.
                #
                # FaSTMM2's raw S11 (mie.f90: (|S1|^2+...+|S4|^2)/2) is the
                # standard Bohren-Huffman dCsca/dOmega = S11/k^2 convention
                # (confirmed empirically: integrating raw S11 over the
                # sphere for a single symmetric sphere reproduces
                # k^2*Csca), not the radiative-transfer phase-function
                # convention (integral over the sphere == 4*pi) that
                # callers actually want -- rescale to that here.
                phase_norm = (
                    4.0 * np.pi / (request.wavenumber**2 * c_sca_vals[i])
                )
                m = result["mueller"][: request.n_theta]
                mueller = [
                    [
                        float(np.degrees(row[1])),
                        float(row[2]) * phase_norm,
                        float(row[3]) * phase_norm,
                    ]
                    for row in m
                ]
        wall_time = time.perf_counter() - t0

        n_solves = len(angles_to_solve)
        raw: dict[str, object] = {
            "c_ext_persolve": c_ext_vals if n_solves > 1 else c_ext_vals[0],
            "c_abs_persolve": c_abs_vals if n_solves > 1 else c_abs_vals[0],
            "c_sca_persolve": c_sca_vals if n_solves > 1 else c_sca_vals[0],
            "asymmetry_persolve": asymmetry_vals if n_solves > 1 else asymmetry_vals[0],
        }
        if directions:
            raw["incidence_angles"] = raw_angles

        return ScatterResult(
            tool="fastmm2",
            backend="python",
            adapter_name=self.name,
            c_ext=sum(c_ext_vals) / n_solves,
            c_abs=sum(c_abs_vals) / n_solves,
            # result["c_sca"] is computed by integrating the far field
            # over the requested angular grid, so it's only as accurate
            # as N_theta/N_phi and converges slowly (confirmed: ~0.62 at
            # N_theta=19/N_phi=1 vs. ~1.39 at N_theta=181/N_phi=32, for a
            # case where the true value is 1.39132). c_ext_minus_c_abs
            # (Cext - Cabs via the optical theorem) is exact regardless
            # of angular resolution and is what MSTM's own Q_sca is
            # analogous to -- use that as the canonical, resolution-
            # independent c_sca for cross-tool comparison.
            c_sca=sum(c_sca_vals) / n_solves,
            asymmetry=sum(asymmetry_vals) / n_solves,
            wall_time_seconds=wall_time,
            n_spheres=request.n_spheres,
            mueller=mueller,
            raw=raw,
        )
