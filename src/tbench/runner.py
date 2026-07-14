"""Run one ClusterRequest (or a wavelength SweepRequest) through a set of
adapters and collect results."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from tbench.adapters.base import ScattererAdapter
from tbench.schema import ClusterRequest, ScatterResult
from tbench.sweep import SweepRequest, expand_sweep


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


def run_sweep(
    sweep: SweepRequest,
    adapters: Sequence[ScattererAdapter],
    progress_callback: Callable[[float, float, str], None] | None = None,
) -> dict[str, list[ScatterResult | None]]:
    """Solve every wavelength in a sweep with every adapter, grouped by
    adapter name (so plotting one line per adapter across wavelengths is
    a direct lookup, not a reshape).

    Unavailable adapters get an all-None result list (same length as the
    wavelength list) rather than being silently dropped, so a dashboard
    always has one entry per adapter it asked for. A solve that raises
    (e.g. GMRES not converging at one particular wavelength) is caught
    and recorded as None for that wavelength only, so one bad point
    doesn't abort the whole sweep.

    progress_callback(fraction_done, wavelength_um, adapter_name) is
    called after each individual solve, if given.
    """
    requests = expand_sweep(sweep)
    results: dict[str, list[ScatterResult | None]] = {a.name: [] for a in adapters}
    total = len(requests) * len(adapters)
    step = 0

    for wl_um, request in zip(sweep.wavelengths_um, requests):
        for adapter in adapters:
            if adapter.is_available():
                try:
                    r = adapter.solve(request)
                except Exception:
                    r = None
            else:
                r = None
            results[adapter.name].append(r)
            step += 1
            if progress_callback is not None:
                progress_callback(step / total, wl_um, adapter.name)

    return results
