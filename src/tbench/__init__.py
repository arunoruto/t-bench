"""tbench -- benchmark harness comparing MSTM and FaSTMM2 on the same
cluster configurations, across both their Python wrappers and CLI binaries.
"""

from tbench.adapters import (
    ALL_ADAPTERS,
    Fastmm2CliAdapter,
    Fastmm2PythonAdapter,
    MstmCliAdapter,
    MstmPythonAdapter,
    ScattererAdapter,
)
from tbench.geometry import load_positions
from tbench.runner import run_benchmark, run_sweep
from tbench.schema import ClusterRequest, ScatterResult
from tbench.sweep import MaterialSpec, SweepRequest, expand_sweep

__all__ = [
    "ClusterRequest",
    "ScatterResult",
    "ScattererAdapter",
    "MstmPythonAdapter",
    "MstmCliAdapter",
    "Fastmm2PythonAdapter",
    "Fastmm2CliAdapter",
    "ALL_ADAPTERS",
    "run_benchmark",
    "run_sweep",
    "load_positions",
    "MaterialSpec",
    "SweepRequest",
    "expand_sweep",
]
__version__ = "0.1.0"
