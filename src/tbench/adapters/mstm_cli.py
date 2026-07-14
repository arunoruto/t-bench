"""MstmCliAdapter -- writes a .inp file, shells out to the standalone
``mstm`` CLI binary, parses its output."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time

from tbench.adapters.base import ScattererAdapter
from tbench.schema import ClusterRequest, ScatterResult


class MstmCliAdapter(ScattererAdapter):
    name = "mstm-cli"

    def __init__(self, binary_path: str = "mstm"):
        self.binary_path = binary_path

    def is_available(self) -> bool:
        return shutil.which(self.binary_path) is not None

    def solve(self, request: ClusterRequest) -> ScatterResult:
        from pymstm._inp import write_inp_file
        from pymstm._parser import parse_mstm_output

        k = request.wavenumber
        scaled_radii = [k * r for r in request.radii]
        scaled_positions = [[k * c for c in p] for p in request.coords]
        ref_re = [ri[0] for ri in request.refractive_index]
        ref_im = [ri[1] for ri in request.refractive_index]
        med_re, med_im = request.medium_refractive_index

        with tempfile.TemporaryDirectory() as tmp:
            inp_path = os.path.join(tmp, "run.inp")
            out_name = "mstm_output.dat"
            write_inp_file(
                inp_path,
                radii=scaled_radii, positions=scaled_positions,
                ref_re=ref_re, ref_im=ref_im,
                medium_ref_re=med_re, medium_ref_im=med_im,
                alpha_deg=request.incident_azimuthal_deg,
                beta_deg=request.incident_polar_deg,
                solution_eps=request.tolerance,
                max_iterations=request.max_iterations,
                calculate_scattering_matrix=False,
                print_sphere_data=False,
                output_file=out_name,
            )
            t0 = time.perf_counter()
            subprocess.run(
                [self.binary_path, inp_path], cwd=tmp, check=True, capture_output=True,
            )
            wall_time = time.perf_counter() - t0
            parsed = parse_mstm_output(os.path.join(tmp, out_name))

        # r_cs is in the same k-scaled coordinate system as scaled_radii --
        # divide back by k before squaring to get a physical cross
        # section, matching MstmPythonAdapter's fix (see its comment).
        r_cs = sum(r**3 for r in scaled_radii) ** (1 / 3)
        area = 3.141592653589793 * (r_cs / k) ** 2
        total = parsed["total"]
        return ScatterResult(
            tool="mstm", backend="cli", adapter_name=self.name,
            c_ext=total["q_ext_unpol"] * area,
            c_abs=total["q_abs_unpol"] * area,
            c_sca=total["q_sca_unpol"] * area,
            wall_time_seconds=wall_time,
            iterations=parsed.get("iterations"),
            n_spheres=request.n_spheres,
            raw=parsed,
        )
