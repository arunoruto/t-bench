from tbench.adapters.base import ScattererAdapter
from tbench.adapters.fastmm2_cli import Fastmm2CliAdapter
from tbench.adapters.fastmm2_python import Fastmm2PythonAdapter
from tbench.adapters.mstm_cli import MstmCliAdapter
from tbench.adapters.mstm_python import MstmPythonAdapter

ALL_ADAPTERS: list[type[ScattererAdapter]] = [
    MstmPythonAdapter, MstmCliAdapter, Fastmm2PythonAdapter, Fastmm2CliAdapter,
]

__all__ = [
    "ScattererAdapter",
    "MstmPythonAdapter",
    "MstmCliAdapter",
    "Fastmm2PythonAdapter",
    "Fastmm2CliAdapter",
    "ALL_ADAPTERS",
]
