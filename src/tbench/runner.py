"""Run one ClusterRequest through a set of adapters and collect results."""

from __future__ import annotations

from collections.abc import Sequence

from tbench.adapters.base import ScattererAdapter
from tbench.schema import ClusterRequest, ScatterResult


def run_benchmark(
    request: ClusterRequest, adapters: Sequence[ScattererAdapter], skip_unavailable: bool = True,
) -> list[ScatterResult]:
    """Solve one request with each adapter in turn, in order given.

    Unavailable adapters (binary not on PATH, package not importable) are
    skipped by default rather than raising, so a benchmark run degrades
    gracefully on a machine that only has some of the tools built.
    """
    results = []
    for adapter in adapters:
        if not adapter.is_available():
            if skip_unavailable:
                continue
            raise RuntimeError(f"Adapter {adapter.name!r} is not available")
        results.append(adapter.solve(request))
    return results
