"""Fastmm2CliAdapter -- writes a geometry.h5, shells out to the standalone
``FaSTMM2`` CLI binary, reads its mueller.h5 output."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import warnings

import numpy as np

from tbench.adapters.base import ScattererAdapter
from tbench.schema import ClusterRequest, ScatterResult


class Fastmm2CliAdapter(ScattererAdapter):
    name = "fastmm2-cli"

    def __init__(self, binary_path: str = "FaSTMM2"):
        self.binary_path = binary_path

    def is_available(self) -> bool:
        if shutil.which(self.binary_path) is None:
            return False
        try:
            import h5py  # noqa: F401
        except ImportError:
            return False
        return True

    def solve(self, request: ClusterRequest) -> ScatterResult:
        import h5py

        if request.medium_refractive_index != (1.0, 0.0):
            warnings.warn(
                "FaSTMM2 has no background-medium model (always vacuum); "
                f"medium_refractive_index={request.medium_refractive_index} is ignored.",
                stacklevel=2,
            )

        n = request.n_spheres
        eps = np.array([complex(*ri) ** 2 for ri in request.refractive_index])
        n_phi = max(1, request.n_phi)

        with tempfile.TemporaryDirectory() as tmp:
            geo_path = os.path.join(tmp, "geometry.h5")
            s_out = os.path.join(tmp, "mueller.h5")
            with h5py.File(geo_path, "w") as fh:
                fh.create_dataset("coord", data=np.asarray(request.coords))
                fh.create_dataset("radius", data=np.asarray(request.radii))
                fh.create_dataset("param_r", data=np.real(eps))
                fh.create_dataset("param_i", data=np.imag(eps))
                fh.create_dataset("tind", data=np.zeros(n, dtype=np.int32))
                fh.create_dataset("angles", data=np.zeros((n, 3)))

            args = [
                self.binary_path, "-geometry_file", geo_path, "-k", str(request.wavenumber),
                "-N_ave", "0", "-N_theta", str(request.n_theta), "-N_phi", str(n_phi),
                "-formulation", str(request.formulation), "-acc", str(request.mlfmm_accuracy),
                "-tol", str(request.tolerance), "-restart", "5",
                "-max_iter", str(request.max_iterations), "-S_out", s_out,
            ]
            t0 = time.perf_counter()
            # NOT capture_output=True: FaSTMM2 prints copious per-iteration
            # diagnostics (every GMRES step, every octree/translation
            # phase), and capture_output buffers all of it in *Python
            # process memory*, unbounded, for the whole run -- confirmed
            # as a real OOM crash on a genuinely hard cluster (the
            # 128-particle fractal aggregate test file) that ground
            # through many iterations before finishing. Neither this
            # adapter nor its caller ever reads .stdout/.stderr (results
            # come from the mueller.h5 output below), so there's no
            # information loss in discarding it outright.
            subprocess.run(
                args, cwd=tmp, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            wall_time = time.perf_counter() - t0

            with h5py.File(s_out) as fh:
                crs = fh["cross_sections"][()]

        # cross_sections is [Cext, Cext-Cabs, Cabs, Csca(far-field
        # integrated), asymmetry]. Cext-Cabs (index 1, optical theorem)
        # is exact regardless of angular resolution; Csca (index 3) is
        # only as accurate as N_theta/N_phi and converges slowly -- see
        # the matching comment in fastmm2_python.py. Use index 1 as the
        # canonical, resolution-independent c_sca for cross-tool comparison.
        return ScatterResult(
            tool="fastmm2", backend="cli", adapter_name=self.name,
            c_ext=float(crs[0]), c_abs=float(crs[2]), c_sca=float(crs[1]),
            asymmetry=float(crs[4]),
            wall_time_seconds=wall_time,
            n_spheres=n,
            raw={"cross_sections": crs.tolist()},
        )
