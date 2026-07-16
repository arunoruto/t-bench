"""MstmPythonAdapter -- drives pymstm.MSTM directly, in-process."""

from __future__ import annotations

import math
import time

from tbench.adapters.base import ScattererAdapter
from tbench.incidence import generate_incidence_directions
from tbench.schema import ClusterRequest, ScatterResult

# Below this size parameter (x = k*r), MSTM crashes the whole process --
# not a catchable Python exception, an actual memory-corruption abort
# (glibc "double free or corruption" / "munmap_chunk(): invalid pointer",
# SIGABRT/SIGSEGV) -- confirmed directly: bisecting a real 128-sphere
# cluster found MSTM solves fine down to x~1.3e-6 but crashes by x~1.3e-7,
# and a synthetic 2-sphere case crashed as high as x~3e-7. FaSTMM2 does
# *not* crash at the same x (returns near-zero, numerically-degenerate but
# valid results instead) -- this is specific to MSTM's own Fortran code,
# not a fundamental property of the physics. Since a process abort can't
# be caught with try/except (it takes the whole interpreter down, which is
# exactly the "dashboard crashes" bug report this fixes), the only
# reliable fix is rejecting the request before it ever reaches MSTM.
# 1e-4 gives ~1000x margin above the observed crash zone, and is already
# so deep in the Rayleigh regime (particle 10,000x smaller than the
# wavelength) that no real use case needs anything smaller -- hitting
# this is essentially always a units/scale mistake, e.g. a cluster file
# scaled into a physically nonsensical size range.
MIN_SIZE_PARAMETER = 1e-4


def check_size_parameters(radii: list[float], wavenumber: float) -> None:
    x_min = wavenumber * min(radii)
    if x_min < MIN_SIZE_PARAMETER:
        raise ValueError(
            f"Size parameter (wavenumber * radius) as low as {x_min:.3g} is "
            f"below MSTM's known-crashing threshold ({MIN_SIZE_PARAMETER:.0e}) "
            "-- MSTM aborts the whole process (not a catchable error) for "
            "pathologically tiny particles relative to the wavelength. This "
            "is almost always a units/scale mistake (e.g. a cluster scaled "
            "into femtometers) -- check that coords/radii and wavelength "
            "are in the same length unit (micrometers)."
        )


class MstmPythonAdapter(ScattererAdapter):
    name = "mstm-python"

    def is_available(self) -> bool:
        try:
            import pymstm  # noqa: F401
        except ImportError:
            return False
        return True

    def solve(self, request: ClusterRequest) -> ScatterResult:
        from pymstm import MSTM

        check_size_parameters(request.radii, request.wavenumber)

        k = request.wavenumber
        scaled_radii = [k * r for r in request.radii]
        scaled_positions = [[k * c for c in p] for p in request.coords]
        orders = [max(4, int(x + 4 * x ** (1 / 3) + 2)) for x in scaled_radii]
        ref_re = [ri[0] for ri in request.refractive_index]
        ref_im = [ri[1] for ri in request.refractive_index]

        mie_val = request.mstm_mie_eps
        if mie_val < 0:
            mie_val = int(mie_val)

        directions = generate_incidence_directions(
            request.n_incidence_angles,
            request.incidence_seed,
        )

        t0 = time.perf_counter()
        c_ext_vals: list[float] = []
        c_abs_vals: list[float] = []
        c_sca_vals: list[float] = []
        total_iters = 0
        raw_angles: list[tuple[float, float]] = []

        m = MSTM()
        try:
            m.set_spheres(
                radii=scaled_radii,
                positions=scaled_positions,
                orders=orders,
                ref_re=ref_re,
                ref_im=ref_im,
            )
            m.set_medium_ref(*request.medium_refractive_index)
            m.set_solver_params(eps=request.tolerance, max_iter=request.max_iterations)
            m.set_mie_eps(mie_val)
            m.set_translation_eps(request.mstm_translation_eps)
            m.set_verbose(False)

            angles_to_solve = (
                directions
                if directions
                else [(request.incident_polar_deg, request.incident_azimuthal_deg)]
            )

            mueller: list[list[float]] | None = None
            for i, (polar_deg, azimuthal_deg) in enumerate(angles_to_solve):
                raw_angles.append((polar_deg, azimuthal_deg))
                m.set_incident(
                    alpha_deg=azimuthal_deg,
                    beta_deg=polar_deg,
                )
                m.prepare()
                raw = m.solve()
                r_cs = m.get_cross_section_radius()

                area = 3.141592653589793 * (r_cs / k) ** 2
                c_ext_vals.append(float(raw["qext_tot"]) * area)
                c_abs_vals.append(float(raw["qabs_tot"]) * area)
                c_sca_vals.append(float(raw["qsca_tot"]) * area)
                total_iters += int(raw["iterations"])

                # S11(theta)/DoLP are orientation-dependent (theta is
                # measured relative to the incident direction), so they
                # don't have a clean meaning averaged across different
                # incidence directions -- only compute them once, from
                # the first solved orientation.
                if request.compute_mueller and i == 0:
                    # get_scattering_angle() re-evaluates the *already
                    # solved* field at a new angle (cheap post-processing,
                    # not a re-solve) -- used in a loop instead of the
                    # faster batch get_scattering_matrix(), which has a
                    # confirmed intermittent memory bug (non-deterministic
                    # garbage/negative S11 across otherwise-identical
                    # runs; see the investigation this session). This
                    # per-angle path matches what pyMSTM's own dashboard
                    # already uses for the same reason.
                    #
                    # get_scattering_angle()'s raw S11/S12 satisfy the
                    # standard Bohren-Huffman convention dCsca/dOmega =
                    # S11/k^2 (confirmed empirically: integrating raw S11
                    # over the full sphere for a single symmetric sphere
                    # reproduces k^2*Csca to within numerical-integration
                    # error), *not* the radiative-transfer phase-function
                    # convention (integral over the sphere == 4*pi) that
                    # is the actual "phase function" callers expect --
                    # rescale here so the reported values are directly
                    # that phase function: p(theta) = 4*pi*S11/(k^2*Csca).
                    phase_norm = 4.0 * math.pi / (k**2 * c_sca_vals[i])
                    phi = math.radians(azimuthal_deg)
                    mueller = []
                    for theta_deg in [
                        180.0 * j / (request.n_theta - 1)
                        for j in range(request.n_theta)
                    ]:
                        costheta = math.cos(math.radians(theta_deg))
                        sm = m.get_scattering_angle(costheta=costheta, phi=phi)
                        mueller.append(
                            [
                                theta_deg,
                                float(sm[0]) * phase_norm,
                                float(sm[1]) * phase_norm,
                            ]
                        )
        finally:
            m.finalize()
        wall_time = time.perf_counter() - t0

        n_solves = len(angles_to_solve)
        c_ext = sum(c_ext_vals) / n_solves
        c_abs = sum(c_abs_vals) / n_solves
        c_sca = sum(c_sca_vals) / n_solves

        raw: dict[str, object] = {
            "c_ext_persolve": c_ext_vals if n_solves > 1 else c_ext_vals[0],
            "c_abs_persolve": c_abs_vals if n_solves > 1 else c_abs_vals[0],
            "c_sca_persolve": c_sca_vals if n_solves > 1 else c_sca_vals[0],
            "total_iterations": total_iters,
        }
        if directions:
            raw["incidence_angles"] = raw_angles

        return ScatterResult(
            tool="mstm",
            backend="python",
            adapter_name=self.name,
            c_ext=c_ext,
            c_abs=c_abs,
            c_sca=c_sca,
            wall_time_seconds=wall_time,
            iterations=total_iters // max(n_solves, 1) if n_solves else None,
            n_spheres=request.n_spheres,
            mueller=mueller,
            raw=raw,
        )
