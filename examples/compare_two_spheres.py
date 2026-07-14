"""Compare MSTM and FaSTMM2 (Python wrapper and CLI, both tools) on the
same two-sphere cluster."""

from tbench import ALL_ADAPTERS, ClusterRequest, run_benchmark

request = ClusterRequest(
    coords=[(-1.5, 0.0, 0.0), (1.5, 0.0, 0.0)],
    radii=[1.0, 1.0],
    refractive_index=[(1.5, 0.01), (1.5, 0.01)],
    wavenumber=1.0,
    n_theta=91,
    n_phi=8,
    tolerance=1e-4,
    max_iterations=2000,
)

adapters = [cls() for cls in ALL_ADAPTERS]
results = run_benchmark(request, adapters)

print(f"{'adapter':16s} {'Cext':>10s} {'Cabs':>10s} {'Csca':>10s} {'time (s)':>10s}")
for r in results:
    print(f"{r.adapter_name:16s} {r.c_ext:10.5f} {r.c_abs:10.5f} {r.c_sca:10.5f} {r.wall_time_seconds:10.4f}")
