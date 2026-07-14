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
from tbench.runner import run_benchmark
from tbench.schema import ClusterRequest, ScatterResult

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
]
__version__ = "0.1.0"
