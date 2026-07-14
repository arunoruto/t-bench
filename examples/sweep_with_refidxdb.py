"""Wavelength sweep, using a real dispersive material from refidxdb,
across all four adapters."""

from tbench import ALL_ADAPTERS, MaterialSpec, SweepRequest, run_sweep

sweep = SweepRequest(
    coords=[(-1.5, 0.0, 0.0), (1.5, 0.0, 0.0)],
    radii=[1.0, 1.0],
    material=MaterialSpec(
        refidxdb_url=(
            "https://refractiveindex.info/database/data/main/SiO2/"
            "nk/Rodriguez-de%20Marcos.yml"
        )
    ),
    wavelengths_um=[0.4, 0.6, 0.8, 1.0],
    n_theta=91,
    n_phi=8,
    tolerance=1e-4,
    max_iterations=2000,
)

adapters = [cls() for cls in ALL_ADAPTERS]
results = run_sweep(sweep, adapters)

print(f"{'wavelength (um)':>16s}", *[f"{name:>16s}" for name in results], sep="  ")
for i, wl in enumerate(sweep.wavelengths_um):
    row = []
    for name in results:
        r = results[name][i]
        row.append(f"{r.c_ext:16.5f}" if r is not None else f"{'FAILED':>16s}")
    print(f"{wl:16.3f}", *row, sep="  ")
