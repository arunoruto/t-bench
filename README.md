# t-bench

Benchmark harness comparing [MSTM](https://github.com/dmckwski/MSTM) and
[FaSTMM2](https://bitbucket.org/planetarysystemresearch/fastmm2) -- two
independent T-matrix codes for electromagnetic scattering from clusters
of spheres -- on the same cluster configurations, across both their
Python wrappers ([pyMSTM](https://github.com/arunoruto/pyMSTM),
[pyFaSTMM](https://github.com/arunoruto/pyFaSTMM)) and their standalone
CLI binaries.

## Architecture

Neither pyMSTM nor pyFaSTMM knows t-bench exists. t-bench depends on
both of them (as local path dependencies -- see `pyproject.toml`'s
`[tool.uv.sources]`); the dependency graph is one-directional and
acyclic. The common interface those two very different tools (and their
CLI binaries, which can't implement a Python class at all) are compared
through lives entirely here, as **adapters**:

```
src/tbench/
  schema.py            ClusterRequest / ScatterResult -- the common
                        pydantic request/result models
  adapters/
    base.py             ScattererAdapter ABC: solve(request) -> result
    mstm_python.py       MstmPythonAdapter    -- calls pymstm.MSTM directly
    mstm_cli.py           MstmCliAdapter       -- writes a .inp, shells out to `mstm`
    fastmm2_python.py     Fastmm2PythonAdapter -- calls pyfastmm.FaSTMM2 directly
    fastmm2_cli.py         Fastmm2CliAdapter    -- writes geometry.h5, shells out to `FaSTMM2`
  runner.py             run_benchmark(request, adapters) -> list[ScatterResult]
```

Each adapter translates the common `ClusterRequest` into that tool's own
native API/file format, and translates the result back into the common
`ScatterResult` (cross sections + wall-clock time + tool-native raw
output for debugging). Adding a third tool later means adding adapters
here -- no changes to any existing tool's own repository.

### Units and conventions

- `coords`/`radii` are in one consistent physical length unit; `wavenumber`
  is `2*pi/wavelength` in the reciprocal of that unit. MSTM natively wants
  pre-scaled size parameters (`x = k*r`) instead of a separate wavenumber
  -- the MSTM adapters do that conversion. FaSTMM2 wants permittivity
  (`eps = m**2`) instead of refractive index -- the FaSTMM2 adapters do
  that conversion.
- `c_sca` is always the resolution-independent value (`Cext - Cabs` via
  the optical theorem for FaSTMM2; MSTM's own `Q_sca` is already this).
  FaSTMM2 also reports a far-field-integrated `Csca` that's only as
  accurate as the requested angular resolution and converges slowly --
  deliberately not used as the canonical value (see the comment in
  `adapters/fastmm2_python.py`), though it's preserved in `raw` for
  reference.
- FaSTMM2 has no background-medium model (always vacuum); a non-trivial
  `medium_refractive_index` is honored by the MSTM adapters and ignored
  (with a warning) by the FaSTMM2 adapters.

## Usage

```python
from tbench import ClusterRequest, ALL_ADAPTERS, run_benchmark

request = ClusterRequest(
    coords=[(-1.5, 0.0, 0.0), (1.5, 0.0, 0.0)],
    radii=[1.0, 1.0],
    refractive_index=[(1.5, 0.01), (1.5, 0.01)],
    wavenumber=1.0,
)

adapters = [cls() for cls in ALL_ADAPTERS]  # each skips itself if unavailable
results = run_benchmark(request, adapters)
for r in results:
    print(r.adapter_name, r.c_ext, r.c_abs, r.c_sca, r.wall_time_seconds)
```

See `examples/compare_two_spheres.py` for a complete runnable example.

## Setup

Requires the `mstm`/`FaSTMM2` CLI binaries on `PATH` (built by
`nix/packages/*/package.nix`, pulled in automatically inside the `devenv`
shell) and `../pyMSTM`/`../pyFaSTMM` checked out as sibling directories
(path dependencies -- see `pyproject.toml`).

```bash
devenv shell   # builds both CLIs and both Python packages
uv run pytest tests/
```

## License

MIT.
