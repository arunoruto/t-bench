"""Load a cluster of spheres from a position file.

Same format pyMSTM's and pyFaSTMM's own dashboards already read (a
PyFracVAL-style whitespace file with x, y, z, radius columns and `#`-
prefixed comment lines, or a plain CSV) -- reimplemented here as a small
standalone utility rather than reaching into either tool's private
_config module, since loading a position file has nothing to do with
either tool's own solver.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt


def load_positions(
    path: str | os.PathLike[str], scale: float = 1.0, gap_factor: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Return an (N, 4) array of [x, y, z, radius].

    Supports whitespace-separated (.dat/.txt/.pos, `#` comment lines
    stripped automatically) and comma-separated (.csv) files.

    *scale* multiplies every column (use to convert the file's own units
    to **micrometers** -- the length unit the rest of a benchmark run
    assumes, matching sweep.py's wavelengths_um). Pick *scale* to convert
    to micrometers specifically, not SI meters: nanometers -> 1e-3,
    angstroms -> 1e-4, millimeters -> 1e3. A common mistake is using the
    nm-to-meters factor (1e-9) instead of nm-to-micrometers (1e-3) --
    that under-scales radii by another 1e-6 and drives the resulting size
    parameter (wavenumber * radius) low enough to crash MSTM (see
    adapters/mstm_python.py's MIN_SIZE_PARAMETER). *gap_factor*
    additionally stretches only the positions (not radii), to separate
    touching spheres if needed.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        data = np.loadtxt(str(path), delimiter=",")
    else:
        data = np.loadtxt(str(path))
    if data.ndim == 1:
        data = data.reshape(-1, 4)
    data = data * scale
    if gap_factor != 1.0:
        data[:, :3] *= gap_factor
    return data
