"""MstmCliAdapter -- writes a .inp file, shells out to the standalone
``mstm`` CLI binary, parses its output."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import time

from tbench.adapters.base import ScattererAdapter
from tbench.adapters.mstm_python import check_size_parameters
from tbench.incidence import generate_incidence_directions
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

        check_size_parameters(request.radii, request.wavenumber)

        k = request.wavenumber
        scaled_radii = [k * r for r in request.radii]
        scaled_positions = [[k * c for c in p] for p in request.coords]
        ref_re = [ri[0] for ri in request.refractive_index]
        ref_im = [ri[1] for ri in request.refractive_index]
        med_re, med_im = request.medium_refractive_index

        mie_val = request.mstm_mie_eps
        if mie_val < 0:
            mie_val = int(mie_val)

        directions = generate_incidence_directions(
            request.n_incidence_angles,
            request.incidence_seed,
        )
        angles_to_solve = (
            directions
            if directions
            else [(request.incident_polar_deg, request.incident_azimuthal_deg)]
        )

        r_cs = sum(r**3 for r in scaled_radii) ** (1 / 3)
        area = math.pi * (r_cs / k) ** 2

        c_ext_vals: list[float] = []
        c_abs_vals: list[float] = []
        c_sca_vals: list[float] = []
        total_iters = 0
        raw_angles: list[tuple[float, float]] = []
        mueller: list[list[float]] | None = None

        t0 = time.perf_counter()
        for i, (polar_deg, azimuthal_deg) in enumerate(angles_to_solve):
            raw_angles.append((polar_deg, azimuthal_deg))
            with tempfile.TemporaryDirectory() as tmp:
                inp_path = os.path.join(tmp, "run.inp")
                out_name = "mstm_output.dat"
                compute_this_mueller = request.compute_mueller and i == 0
                write_inp_file(
                    inp_path,
                    radii=scaled_radii,
                    positions=scaled_positions,
                    ref_re=ref_re,
                    ref_im=ref_im,
                    medium_ref_re=med_re,
                    medium_ref_im=med_im,
                    alpha_deg=azimuthal_deg,
                    beta_deg=polar_deg,
                    solution_eps=request.tolerance,
                    max_iterations=request.max_iterations,
                    mie_eps=float(mie_val),
                    translation_eps=request.mstm_translation_eps,
                    calculate_scattering_matrix=compute_this_mueller,
                    normalize_s11=False,
                    print_sphere_data=False,
                    output_file=out_name,
                )
                subprocess.run(
                    [self.binary_path, inp_path],
                    cwd=tmp,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                parsed = parse_mstm_output(os.path.join(tmp, out_name))

            total = parsed["total"]
            c_ext_vals.append(total["q_ext_unpol"] * area)
            c_abs_vals.append(total["q_abs_unpol"] * area)
            c_sca_vals.append(total["q_sca_unpol"] * area)
            it = parsed.get("iterations")
            if isinstance(it, (int, float)):
                total_iters += int(it)

            if compute_this_mueller and parsed.get("scattering_matrix"):
                sm = parsed["scattering_matrix"]
                # The CLI's own text table always spans -180..+180 (361
                # points at 1deg resolution -- scattering_map_dimension
                # has no effect on this particular output, confirmed
                # empirically), where negative angle labels are theta
                # paired with the *opposite* azimuthal half-plane
                # (phi=alpha+180deg) rather than literal negative angles
                # -- same convention pyMSTM's own dashboard documents.
                # Keep only the standard 0-180deg, phi=alpha cut.
                # matrix columns are column-major (S11, S21, S31, S41,
                # S12, ...) -- see the historical bug-hunt comment in
                # pymstm/_parser.py's _SM_LABELS; S11 is index 0, S12 is
                # index 4, not 1.
                #
                # normalize_s11=False (above) gets this path onto the same
                # *shape* as get_scattering_angle()/FaSTMM2's raw
                # convention, but leaves a residual, exactly-constant 2*pi
                # factor across every angle (confirmed empirically against
                # mstm-python's raw S11 at theta=90 on an identical case:
                # raw/(2*pi)/mstm-python = 1.0000 to 5 sig figs) -- divide
                # it out here rather than in pymstm itself, since the .inp
                # text writer has no further "true unnormalized" mode to
                # ask for.
                #
                # That raw (post-2*pi-correction) S11/S12 still follows the
                # Bohren-Huffman dCsca/dOmega = S11/k^2 convention (confirmed
                # empirically: integrating raw S11 over the sphere for a
                # single symmetric sphere reproduces k^2*Csca), not the
                # radiative-transfer phase-function convention (integral
                # over the sphere == 4*pi) callers actually want -- apply
                # that rescaling here too, same as the other three adapters.
                phase_norm = 4.0 * math.pi / (k**2 * c_sca_vals[i])
                mueller = [
                    [
                        theta,
                        row[0] / (2 * math.pi) * phase_norm,
                        row[4] / (2 * math.pi) * phase_norm,
                    ]
                    for theta, row in zip(sm["angles_deg"], sm["matrix"])
                    if theta >= 0
                ]
        wall_time = time.perf_counter() - t0

        n_solves = len(angles_to_solve)
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
            backend="cli",
            adapter_name=self.name,
            c_ext=sum(c_ext_vals) / n_solves,
            c_abs=sum(c_abs_vals) / n_solves,
            c_sca=sum(c_sca_vals) / n_solves,
            wall_time_seconds=wall_time,
            iterations=total_iters // max(n_solves, 1) if n_solves else None,
            n_spheres=request.n_spheres,
            mueller=mueller,
            raw=raw,
        )
