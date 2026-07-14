"""
The common solver interface -- implemented by adapters *in this project*,
not by pymstm's MSTM class or pyfastmm's FaSTMM2 class.

Why here and not there: a CLI binary can't subclass a Python ABC at all,
and this benchmark explicitly wants to compare CLI binaries alongside the
Python wrappers. So the shared interface can only ever be implemented by
something that wraps each tool from the outside -- these adapters. That
also means pymstm and pyfastmm never need to import or know about
t-bench, which is what keeps the dependency graph acyclic: t-bench
depends on pymstm and pyfastmm (see pyproject.toml), never the reverse.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from tbench.schema import ClusterRequest, ScatterResult


class ScattererAdapter(ABC):
    """One way of computing a ClusterRequest's scattering response --
    e.g. "MSTM via its Python wrapper" or "FaSTMM2 via its CLI binary"."""

    name: str

    @abstractmethod
    def solve(self, request: ClusterRequest) -> ScatterResult:
        """Solve one request, returning cross sections plus timing."""

    def is_available(self) -> bool:
        """Whether this adapter can actually run right now (binary on
        PATH, package importable, etc.) -- checked before solve() so a
        benchmark run can skip unavailable adapters instead of crashing."""
        return True
