"""MstmPythonAdapter -- drives pymstm.MSTM directly, in-process."""

from __future__ import annotations

import time

from tbench.adapters.base import ScattererAdapter
from tbench.schema import ClusterRequest, ScatterResult


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
