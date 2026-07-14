"""Fastmm2PythonAdapter -- drives pyfastmm.FaSTMM2 directly, in-process."""

from __future__ import annotations

import time
import warnings

from tbench.adapters.base import ScattererAdapter
from tbench.schema import ClusterRequest, ScatterResult


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

        eps = [complex(*ri) ** 2 for ri in request.refractive_index]

        t0 = time.perf_counter()
        f = FaSTMM2()
        result = f.solve(
            request.coords, request.radii, eps, request.wavenumber,
            N_theta=request.n_theta, N_phi=max(1, request.n_phi),
            formulation=request.formulation, acc=request.mlfmm_accuracy,
            tol=request.tolerance, restart=5, max_iter=request.max_iterations,
        )
        wall_time = time.perf_counter() - t0

        return ScatterResult(
            tool="fastmm2", backend="python", adapter_name=self.name,
            c_ext=result["c_ext"], c_abs=result["c_abs"],
            # result["c_sca"] is computed by integrating the far field
            # over the requested angular grid, so it's only as accurate
            # as N_theta/N_phi and converges slowly (confirmed: ~0.62 at
            # N_theta=19/N_phi=1 vs. ~1.39 at N_theta=181/N_phi=32, for a
            # case where the true value is 1.39132). c_ext_minus_c_abs
            # (Cext - Cabs via the optical theorem) is exact regardless
            # of angular resolution and is what MSTM's own Q_sca is
            # analogous to -- use that as the canonical, resolution-
            # independent c_sca for cross-tool comparison.
            c_sca=result["c_ext_minus_c_abs"],
            asymmetry=result["asymmetry"],
            wall_time_seconds=wall_time,
            n_spheres=request.n_spheres,
            raw={k: v for k, v in result.items() if k not in ("mueller", "jones")},
        )
