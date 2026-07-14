"""MstmPythonAdapter -- drives pymstm.MSTM directly, in-process."""

from __future__ import annotations

import time

from tbench.adapters.base import ScattererAdapter
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

        # MSTM wants size parameters (x = k*r) directly, not a separate
        # wavenumber -- see schema.py's module docstring.
        k = request.wavenumber
        scaled_radii = [k * r for r in request.radii]
        scaled_positions = [[k * c for c in p] for p in request.coords]
        # "A good default is max(4, int(x + 4*x**(1/3) + 2))" -- MSTM's
        # own set_spheres() docstring, x = the size parameter.
        orders = [max(4, int(x + 4 * x ** (1 / 3) + 2)) for x in scaled_radii]
        ref_re = [ri[0] for ri in request.refractive_index]
        ref_im = [ri[1] for ri in request.refractive_index]

        t0 = time.perf_counter()
        m = MSTM()
        try:
            m.set_spheres(
                radii=scaled_radii, positions=scaled_positions, orders=orders,
                ref_re=ref_re, ref_im=ref_im,
            )
            m.set_medium_ref(*request.medium_refractive_index)
            m.set_incident(
                alpha_deg=request.incident_azimuthal_deg,
                beta_deg=request.incident_polar_deg,
            )
            m.set_solver_params(eps=request.tolerance, max_iter=request.max_iterations)
            m.set_verbose(False)
            m.prepare()
            raw = m.solve()
            r_cs = m.get_cross_section_radius()
        finally:
            m.finalize()
        wall_time = time.perf_counter() - t0

        # r_cs is in the same k-scaled (size-parameter) coordinate system
        # as scaled_radii/scaled_positions -- Q_ext etc. are dimensionless
        # efficiencies (fine as-is), but converting to a *physical* cross
        # section needs r_cs divided back by k first. Confirmed the bug
        # this fixes: without the /k, a k=1.0 sanity case looks right by
        # coincidence (dividing by 1 is a no-op) but a k=12.57 case (0.5um
        # wavelength, 1um-radius spheres) was off by a factor of k**2
        # (~158x), producing Cext=2632 instead of the ~16 FaSTMM2 computes
        # for the identical physical problem.
        area = 3.141592653589793 * (r_cs / k) ** 2
        return ScatterResult(
            tool="mstm", backend="python", adapter_name=self.name,
            c_ext=float(raw["qext_tot"]) * area,
            c_abs=float(raw["qabs_tot"]) * area,
            c_sca=float(raw["qsca_tot"]) * area,
            wall_time_seconds=wall_time,
            iterations=int(raw["iterations"]),
            n_spheres=request.n_spheres,
            raw={k2: (v.tolist() if hasattr(v, "tolist") else v) for k2, v in raw.items()},
        )
