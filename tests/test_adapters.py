"""Adapter tests -- each skips automatically if its tool isn't available
(binary not on PATH, or the Python package failed to build)."""

import pytest

from tbench.adapters import (
    Fastmm2CliAdapter,
    Fastmm2PythonAdapter,
    MstmCliAdapter,
    MstmPythonAdapter,
)
from tbench.adapters.mstm_python import MIN_SIZE_PARAMETER, check_size_parameters
from tbench.schema import ClusterRequest

_TWO_SPHERES = ClusterRequest(
    coords=[(-1.5, 0.0, 0.0), (1.5, 0.0, 0.0)],
    radii=[1.0, 1.0],
    refractive_index=[(1.5, 0.01), (1.5, 0.01)],
    wavenumber=1.0,
    n_theta=19,
    n_phi=1,
    tolerance=1e-4,
    max_iterations=500,
)

_ADAPTER_CLASSES = [MstmPythonAdapter, MstmCliAdapter, Fastmm2PythonAdapter, Fastmm2CliAdapter]


@pytest.mark.parametrize("adapter_cls", _ADAPTER_CLASSES)
def test_adapter_solves_two_spheres(adapter_cls):
    adapter = adapter_cls()
    if not adapter.is_available():
        pytest.skip(f"{adapter.name} not available")

    result = adapter.solve(_TWO_SPHERES)

    assert result.adapter_name == adapter.name
    assert result.n_spheres == 2
    assert result.c_ext > 0
    assert result.c_abs > 0
    assert result.c_sca > 0
    assert result.wall_time_seconds > 0
    # Optical theorem: Cext = Cabs + Csca, for both tools' canonical c_sca
    # (see the comment in adapters/fastmm2_python.py for why FaSTMM2's
    # far-field-integrated Csca is deliberately not used here).
    assert result.c_ext == pytest.approx(result.c_abs + result.c_sca, rel=1e-3)


def test_all_available_adapters_agree_on_cross_sections():
    """Two independent T-matrix implementations, on the same physical
    problem, should agree on cross sections to a few significant figures
    -- this is as much a cross-validation of MSTM against FaSTMM2 as it
    is a test of the adapters themselves."""
    results = []
    for adapter_cls in _ADAPTER_CLASSES:
        adapter = adapter_cls()
        if adapter.is_available():
            results.append(adapter.solve(_TWO_SPHERES))

    if len(results) < 2:
        pytest.skip("need at least two available adapters to compare")

    c_ext_values = [r.c_ext for r in results]
    spread = (max(c_ext_values) - min(c_ext_values)) / max(c_ext_values)
    assert spread < 0.01, f"Cext disagreement too large: {dict(zip((r.adapter_name for r in results), c_ext_values))}"


def test_mstm_matches_fastmm2_at_nontrivial_wavenumber():
    """Regression test for a real bug: MstmPythonAdapter/MstmCliAdapter
    convert MSTM's dimensionless efficiencies to physical cross sections
    via a cross-section radius that MSTM reports in the same k-scaled
    (size-parameter) coordinate system it was given (scaled_radii = k *
    request.radii) -- forgetting to divide that radius back by k before
    squaring made cross sections wrong by a factor of k**2 for any
    k != 1.0. The bug hid in test_all_available_adapters_agree_on_cross_sections
    above because that test happens to use k=1.0 (a no-op division).
    Found via a real SweepRequest at k=2*pi/0.5um=12.566 (1um-radius
    spheres at 0.5um wavelength), where MSTM's Cext came out ~160x
    FaSTMM2's for the identical physical problem -- same radii, same
    wavenumber, only the *conversion* to a physical cross section was
    wrong. Same two spheres as the other tests here, just at k != 1.0.
    """
    mstm = MstmPythonAdapter()
    fastmm2 = Fastmm2PythonAdapter()
    if not (mstm.is_available() and fastmm2.is_available()):
        pytest.skip("need both mstm-python and fastmm2-python available")

    request = ClusterRequest(
        coords=[(-1.5, 0.0, 0.0), (1.5, 0.0, 0.0)], radii=[1.0, 1.0],
        refractive_index=[(1.5, 0.01), (1.5, 0.01)],
        wavenumber=2 * 3.141592653589793 / 0.5,  # k for a 0.5um wavelength
        n_theta=19, n_phi=1, tolerance=1e-4, max_iterations=500,
    )

    r_mstm = mstm.solve(request)
    r_fastmm2 = fastmm2.solve(request)
    assert r_mstm.c_ext == pytest.approx(r_fastmm2.c_ext, rel=0.05)


def test_check_size_parameters_rejects_pathologically_small_particles():
    """Regression test for a real bug: MSTM aborts the whole process (a
    memory-corruption SIGABRT/SIGSEGV, not a catchable exception) when
    given a size parameter (x = wavenumber * radius) far enough below 1.
    Found via a real dashboard crash report -- a user scaled a cluster's
    positions/radii into nanometers (scale=1e-9) while wavelengths stayed
    in micrometers, producing x as low as ~1e-8, which crashed the whole
    process. Confirmed by bisecting the real 128-sphere test cluster: MSTM
    solves fine down to x~1.3e-6 but crashes by x~1.3e-7. This guard must
    reject such requests with a plain ValueError *before* MSTM ever sees
    them, since nothing downstream can catch a process abort."""
    with pytest.raises(ValueError, match="below MSTM's known-crashing threshold"):
        check_size_parameters(radii=[1.0], wavenumber=MIN_SIZE_PARAMETER / 10)

    # Just above the threshold should pass silently.
    check_size_parameters(radii=[1.0], wavenumber=MIN_SIZE_PARAMETER * 10)


@pytest.mark.parametrize("adapter_cls", [MstmPythonAdapter, MstmCliAdapter])
def test_mstm_adapter_rejects_pathologically_small_particles(adapter_cls):
    adapter = adapter_cls()
    if not adapter.is_available():
        pytest.skip(f"{adapter.name} not available")

    request = ClusterRequest(
        coords=[(-1.5, 0.0, 0.0), (1.5, 0.0, 0.0)],
        radii=[1e-9, 1e-9],
        refractive_index=[(1.5, 0.01), (1.5, 0.01)],
        wavenumber=1.0,
        n_theta=19, n_phi=1, tolerance=1e-4, max_iterations=500,
    )
    with pytest.raises(ValueError, match="below MSTM's known-crashing threshold"):
        adapter.solve(request)
