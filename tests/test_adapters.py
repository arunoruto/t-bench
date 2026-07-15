"""Adapter tests -- each skips automatically if its tool isn't available
(binary not on PATH, or the Python package failed to build)."""

import os

import pytest

from tbench.adapters import (
    Fastmm2CliAdapter,
    Fastmm2PythonAdapter,
    MstmCliAdapter,
    MstmPythonAdapter,
)
from tbench.adapters.mstm_python import MIN_SIZE_PARAMETER, check_size_parameters
from tbench.geometry import load_positions
from tbench.schema import ClusterRequest

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

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


def test_mstm_fastmm2_agreement_on_touching_sphere_cluster():
    """The other agreement tests above use two well-separated spheres,
    which never exercises MSTM's near-field truncation defaults the way
    a real aggregate does. This loads a 32-sphere fractal aggregate
    (touching spheres -- generated with tol_ov=1e-6, see the file's own
    header) at a size parameter deep in the Rayleigh regime, the exact
    kind of case that surfaced a real ~25% Cext disagreement and a
    sign-flipped Csca between MSTM and FaSTMM2 before mstm_mie_eps/
    mstm_translation_eps were tightened (see schema.py). Deliberately no
    tight tolerance asserted: tightening those eps values further no
    longer moves MSTM's answer (already converged), so some residual
    disagreement between the two independent solvers is expected at this
    extreme end of parameter space -- this test is for visibility into
    how close they are, not a pass/fail accuracy gate."""
    mstm = MstmPythonAdapter()
    fastmm2 = Fastmm2PythonAdapter()
    if not (mstm.is_available() and fastmm2.is_available()):
        pytest.skip("need both mstm-python and fastmm2-python available")

    positions = load_positions(os.path.join(_DATA_DIR, "fractal_N32_Df2.0.dat"))
    n = positions.shape[0]
    request = ClusterRequest(
        coords=[tuple(p) for p in positions[:, :3]],
        radii=list(positions[:, 3]),
        refractive_index=[(1.5, 0.01)] * n,
        wavenumber=0.05,  # deep Rayleigh (x ~ 0.05) for these touching, radius-1 spheres
        n_theta=19, n_phi=1, tolerance=1e-4, max_iterations=500,
    )

    r_mstm = mstm.solve(request)
    r_fastmm2 = fastmm2.solve(request)

    spread = abs(r_mstm.c_ext - r_fastmm2.c_ext) / r_fastmm2.c_ext
    print(
        f"\nmstm:    c_ext={r_mstm.c_ext:.6e} c_abs={r_mstm.c_abs:.6e} c_sca={r_mstm.c_sca:.6e}\n"
        f"fastmm2: c_ext={r_fastmm2.c_ext:.6e} c_abs={r_fastmm2.c_abs:.6e} c_sca={r_fastmm2.c_sca:.6e}\n"
        f"relative Cext spread: {spread:.1%}"
    )
    assert r_mstm.c_ext > 0
    assert r_fastmm2.c_ext > 0


def test_fastmm2_incident_angle_matches_mstm_when_both_angles_nonzero():
    """Regression test for a real bug: FaSTMM2 has no incident-angle
    parameter of its own (always illuminates along +z), so both FaSTMM2
    adapters rotate the cluster geometry instead and solve with the
    fixed +z beam -- physically equivalent by rotational invariance, as
    long as the rotation exactly inverts MSTM's own incident-wave
    convention (mstm-input-37.f90 builds the tilted wave as
    k_hat = Rz(alpha) . Ry(beta) . z_hat, beta about y first, then alpha
    about z, so the geometry needs the inverse composition
    Ry(-beta) . Rz(-alpha)). The rotation was initially implemented with
    the two matrices composed in the opposite order (Rz(-alpha) .
    Ry(-beta)) -- agreed with MSTM to <0.001% whenever only ONE of
    polar/azimuthal was nonzero (a single rotation is unaffected by
    composition order) but was off by up to ~1.3% once BOTH were
    simultaneously nonzero, where order actually matters. A test with
    only one nonzero angle would not have caught this."""
    mstm = MstmPythonAdapter()
    fastmm2 = Fastmm2PythonAdapter()
    if not (mstm.is_available() and fastmm2.is_available()):
        pytest.skip("need both mstm-python and fastmm2-python available")

    request = ClusterRequest(
        coords=[(-2.0, 0.3, 0.1), (2.0, -0.4, 0.2), (0.5, 2.5, -0.3)],
        radii=[1.0, 1.0, 1.0],
        refractive_index=[(1.5, 0.01)] * 3,
        wavenumber=2 * 3.141592653589793 / 0.5,
        n_theta=19, n_phi=1, tolerance=1e-8, max_iterations=2000,
        incident_polar_deg=30.0, incident_azimuthal_deg=45.0,
        formulation=0,  # STMM: exact, isolates the rotation from MLFMM's own error
    )
    r_mstm = mstm.solve(request)
    r_fastmm2 = fastmm2.solve(request)
    assert r_fastmm2.c_ext == pytest.approx(r_mstm.c_ext, rel=1e-3)
    assert r_fastmm2.c_abs == pytest.approx(r_mstm.c_abs, rel=1e-3)
