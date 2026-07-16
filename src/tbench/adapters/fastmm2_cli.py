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
from tbench.adapters.fastmm2_python import _rotate_positions_for_incident
from tbench.incidence import generate_incidence_directions
from tbench.schema import ClusterRequest, ScatterResult


class Fastmm2CliAdapter(ScattererAdapter):
    name = "fastmm2-cli"

    def __init__(
        self, binary_path: str = "FaSTMM2", omp_num_threads: int | None = None
    ):
        self.binary_path = binary_path
        # FaSTMM2 is built with -fopenmp (see external/fastmm2/src/CMakeLists.txt);
        # None leaves OMP_NUM_THREADS unset, i.e. OpenMP's own default (usually
        # all visible cores).
        self.omp_num_threads = omp_num_threads

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

        directions = generate_incidence_directions(
            request.n_incidence_angles,
            request.incidence_seed,
        )
        angles_to_solve = (
            directions
            if directions
            else [(request.incident_polar_deg, request.incident_azimuthal_deg)]
        )

        n = request.n_spheres
        eps = np.array([complex(*ri) ** 2 for ri in request.refractive_index])
        n_phi = max(1, request.n_phi)

        c_ext_vals: list[float] = []
        c_abs_vals: list[float] = []
        c_sca_vals: list[float] = []
        asymmetry_vals: list[float] = []
        raw_angles: list[tuple[float, float]] = []
        mueller: list[list[float]] | None = None

        t0 = time.perf_counter()
        for i, (polar_deg, azimuthal_deg) in enumerate(angles_to_solve):
            raw_angles.append((polar_deg, azimuthal_deg))
            coords = _rotate_positions_for_incident(
                request.coords,
                polar_deg,
                azimuthal_deg,
            )
            with tempfile.TemporaryDirectory() as tmp:
                geo_path = os.path.join(tmp, "geometry.h5")
                s_out = os.path.join(tmp, "mueller.h5")
                with h5py.File(geo_path, "w") as fh:
                    fh.create_dataset("coord", data=np.asarray(coords))
                    fh.create_dataset("radius", data=np.asarray(request.radii))
                    fh.create_dataset("param_r", data=np.real(eps))
                    fh.create_dataset("param_i", data=np.imag(eps))
                    fh.create_dataset("tind", data=np.zeros(n, dtype=np.int32))
                    fh.create_dataset("angles", data=np.zeros((n, 3)))

                args = [
                    self.binary_path,
                    "-geometry_file",
                    geo_path,
                    "-k",
                    str(request.wavenumber),
                    "-N_ave",
                    "0",
                    "-N_theta",
                    str(request.n_theta),
                    "-N_phi",
                    str(n_phi),
                    "-formulation",
                    str(request.formulation),
                    "-acc",
                    str(request.mlfmm_accuracy),
                    "-tol",
                    str(request.tolerance),
                    "-restart",
                    "5",
                    "-max_iter",
                    str(request.max_iterations),
                    "-S_out",
                    s_out,
                ]
                env_override = None
                if self.omp_num_threads is not None:
                    env_override = {
                        **os.environ,
                        "OMP_NUM_THREADS": str(self.omp_num_threads),
                    }
                subprocess.run(
                    args,
                    cwd=tmp,
                    check=True,
                    env=env_override,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                raw_mueller = None
                with h5py.File(s_out) as fh:
                    crs = fh["cross_sections"][()]
                    if request.compute_mueller and i == 0 and "mueller" in fh:
                        # write2file_mueller (io.f90) writes the Fortran
                        # array's own (n_angles, 18) shape via HDF5 dims
                        # taken straight from size(A,1)/size(A,2) with no
                        # row/column-major correction, so h5py reads it
                        # back transposed as (18, n_angles) -- confirmed
                        # directly (theta column came back as literal
                        # degrees in the hundreds of thousands before
                        # transposing). .T recovers the same
                        # [phi, theta(radians), S11, S12, ...] layout as
                        # the Python binding's result["mueller"] -- see
                        # fastmm2_python.py's comment. First n_theta rows
                        # are the phi=0 cut regardless of N_phi.
                        raw_mueller = fh["mueller"][()].T[: request.n_theta]

            # cross_sections is [Cext, Cext-Cabs, Cabs, Csca(far-field
            # integrated), asymmetry] -- see fastmm2_python.py's matching
            # comment for why index 1 (optical theorem) is used as c_sca
            # instead of index 3 (far-field integrated, resolution-
            # dependent).
            c_ext_vals.append(float(crs[0]))
            c_abs_vals.append(float(crs[2]))
            c_sca_vals.append(float(crs[1]))
            asymmetry_vals.append(float(crs[4]))
            if raw_mueller is not None:
                mueller = [
                    [float(np.degrees(row[1])), float(row[2]), float(row[3])]
                    for row in raw_mueller
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
            backend="cli",
            adapter_name=self.name,
            c_ext=sum(c_ext_vals) / n_solves,
            c_abs=sum(c_abs_vals) / n_solves,
            c_sca=sum(c_sca_vals) / n_solves,
            asymmetry=sum(asymmetry_vals) / n_solves,
            wall_time_seconds=wall_time,
            n_spheres=n,
            mueller=mueller,
            raw=raw,
        )
