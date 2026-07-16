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

_ADAPTER_CLASSES = [
    MstmPythonAdapter,
    MstmCliAdapter,
    Fastmm2PythonAdapter,
    Fastmm2CliAdapter,
]


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
    # Regression test: both FaSTMM2 adapters briefly hardcoded
    # asymmetry=None unconditionally (even for a single, non-averaged
    # solve) while adding incidence-angle averaging -- silently dropping
    # data FaSTMM2 always computes. MSTM's own adapters never report this
    # at all, so only check the tools that are supposed to have it.
    if result.tool == "fastmm2":
        assert result.asymmetry is not None


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
    assert spread < 0.01, (
        f"Cext disagreement too large: {dict(zip((r.adapter_name for r in results), c_ext_values))}"
    )


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
        coords=[(-1.5, 0.0, 0.0), (1.5, 0.0, 0.0)],
        radii=[1.0, 1.0],
        refractive_index=[(1.5, 0.01), (1.5, 0.01)],
        wavenumber=2 * 3.141592653589793 / 0.5,  # k for a 0.5um wavelength
        n_theta=19,
        n_phi=1,
        tolerance=1e-4,
        max_iterations=500,
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
        n_theta=19,
        n_phi=1,
        tolerance=1e-4,
        max_iterations=500,
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
        n_theta=19,
        n_phi=1,
        tolerance=1e-4,
        max_iterations=500,
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
        n_theta=19,
        n_phi=1,
        tolerance=1e-8,
        max_iterations=2000,
        incident_polar_deg=30.0,
        incident_azimuthal_deg=45.0,
        formulation=0,  # STMM: exact, isolates the rotation from MLFMM's own error
    )
    r_mstm = mstm.solve(request)
    r_fastmm2 = fastmm2.solve(request)
    assert r_fastmm2.c_ext == pytest.approx(r_mstm.c_ext, rel=1e-3)
    assert r_fastmm2.c_abs == pytest.approx(r_mstm.c_abs, rel=1e-3)


def test_incidence_averaging_same_angles_on_both_tools():
    """Both adapters must receive identical (polar, azimuthal) pairs from
    the shared ``generate_incidence_directions()`` call -- same seed + same
    N ensures a deterministic, reproducible sequence.  If this ever fails,
    incidence-averaged results are not comparable."""
    mstm = MstmPythonAdapter()
    fastmm2 = Fastmm2PythonAdapter()
    if not (mstm.is_available() and fastmm2.is_available()):
        pytest.skip("need both mstm-python and fastmm2-python available")

    request = ClusterRequest(
        coords=[(-1.5, 0.0, 0.0), (1.5, 0.0, 0.0)],
        radii=[1.0, 1.0],
        refractive_index=[(1.5, 0.01), (1.5, 0.01)],
        wavenumber=1.0,
        n_theta=19,
        n_phi=1,
        tolerance=1e-4,
        max_iterations=500,
        n_incidence_angles=7,
        incidence_seed=42,
        formulation=0,  # STMM: isolate incidence averaging from MLFMM
    )
    r_mstm = mstm.solve(request)
    r_fastmm2 = fastmm2.solve(request)

    assert "incidence_angles" in r_mstm.raw
    assert "incidence_angles" in r_fastmm2.raw
    assert r_mstm.raw["incidence_angles"] == r_fastmm2.raw["incidence_angles"]
    assert len(r_mstm.raw["incidence_angles"]) == 7
    # Averaged cross sections should be positive
    assert r_mstm.c_sca > 0
    assert r_fastmm2.c_sca > 0


def test_mueller_matrix_agrees_across_all_adapters():
    """Regression test for three real bugs found while building the
    compute_mueller feature, one per adapter:

    - mstm-cli: the .inp ``scattering_map_dimension`` keyword was
      initially assumed to control the CLI's angular resolution, but was
      confirmed empirically to have zero effect on the "scattering matrix
      in incident plane" text table -- it's always -180..180deg at 1deg
      resolution (361 points) regardless. mstm-cli therefore has its own
      native grid, filtered here to the 0..180deg half. Separately, MSTM's
      normalize_s11 keyword (default true) scales S11 to an internal
      convention; setting it false plus dividing by a residual, exactly-
      constant 2*pi factor (confirmed via mstm-python's raw S11 at
      theta=90: raw/(2*pi)/mstm-python == 1.0000 to 5 sig figs) recovers
      the same *magnitude* convention as get_scattering_angle()/FaSTMM2 --
      not just the same angular shape.
    - fastmm2-cli: mueller.h5's "mueller" dataset is written by Fortran's
      write2file_mueller (io.f90) using HDF5 dims taken straight from the
      array's own size(A,1)/size(A,2), with no row/column-major
      correction -- h5py silently reads it back transposed, so it needs
      a .T to be usable at all.
    - fastmm2-python (and cli): both use FaSTMM2's [phi, theta(radians),
      S11, S12, ...] mueller layout, phi outermost/theta innermost, so
      the first n_theta rows are always the phi=0 cut regardless of N_phi.

    This test would have caught all three: the shape assertions catch a
    non-transposed/mis-shaped array, and the direct S11 magnitude
    cross-check against mstm-python (not just DoLP, which cancels any
    overall S11 scale factor) catches the normalize_s11/2*pi bug that a
    DoLP-only check would silently pass.
    """
    mstm_py = MstmPythonAdapter()
    mstm_cli = MstmCliAdapter()
    fastmm2_py = Fastmm2PythonAdapter()
    fastmm2_cli = Fastmm2CliAdapter()
    adapters = {
        "mstm-python": mstm_py,
        "mstm-cli": mstm_cli,
        "fastmm2-python": fastmm2_py,
        "fastmm2-cli": fastmm2_cli,
    }
    available = {n: a for n, a in adapters.items() if a.is_available()}
    if len(available) < 2:
        pytest.skip("need at least two Mueller-capable adapters available")

    request = ClusterRequest(
        coords=[(-1.5, 0.0, 0.0), (1.5, 0.0, 0.0)],
        radii=[1.0, 1.0],
        refractive_index=[(1.5, 0.01), (1.5, 0.01)],
        wavenumber=2 * 3.141592653589793 / 0.5,
        n_theta=21,
        n_phi=1,
        tolerance=1e-8,
        max_iterations=2000,
        formulation=0,  # STMM: exact, no MLFMM angular error to muddy the comparison
        compute_mueller=True,
    )

    results = {name: adapter.solve(request) for name, adapter in available.items()}
    for name, r in results.items():
        assert r.mueller is not None, f"{name} did not report a mueller matrix"
        assert len(r.mueller) > 0, f"{name} reported an empty mueller matrix"
        for theta_deg, s11, _s12 in r.mueller:
            assert 0.0 <= theta_deg <= 180.0
            assert s11 >= 0, f"{name}: negative S11 at theta={theta_deg}"

    # n_theta-driven adapters (everything but mstm-cli, which has its own
    # native 181-point grid) must land on identical theta grids and agree
    # on S11/DoLP at every angle.
    grid_adapters = {n: r for n, r in results.items() if n != "mstm-cli"}
    if len(grid_adapters) >= 2:
        names = list(grid_adapters)
        ref_theta = [row[0] for row in grid_adapters[names[0]].mueller]
        for name in names[1:]:
            theta = [row[0] for row in grid_adapters[name].mueller]
            assert theta == pytest.approx(ref_theta, abs=1e-6)

        for i in range(len(ref_theta)):
            s11 = {n: grid_adapters[n].mueller[i][1] for n in names}
            s12 = {n: grid_adapters[n].mueller[i][2] for n in names}
            ref = names[0]
            for name in names[1:]:
                assert s11[name] == pytest.approx(s11[ref], rel=0.02)
                dolp_ref = -s12[ref] / s11[ref]
                dolp_other = -s12[name] / s11[name]
                assert dolp_other == pytest.approx(dolp_ref, abs=0.02)

    # mstm-cli has its own native grid, so cross-check magnitude (not just
    # shape) against mstm-python at matching integer-degree angles --
    # this is what catches the normalize_s11 + residual-2*pi bug.
    if "mstm-cli" in results and "mstm-python" in results:
        cli_mueller = results["mstm-cli"].mueller
        py_mueller = results["mstm-python"].mueller
        cli_theta = [row[0] for row in cli_mueller]
        for theta_deg, py_s11, _py_s12 in py_mueller:
            idx = min(
                range(len(cli_theta)), key=lambda j: abs(cli_theta[j] - theta_deg)
            )
            cli_s11 = cli_mueller[idx][1]
            assert cli_s11 == pytest.approx(py_s11, rel=0.02), (
                f"mstm-cli S11 magnitude mismatch at theta={theta_deg}"
            )


@pytest.mark.parametrize(
    "adapter_cls",
    [MstmPythonAdapter, MstmCliAdapter, Fastmm2PythonAdapter, Fastmm2CliAdapter],
)
def test_mueller_s11_is_normalized_phase_function(adapter_cls):
    """Regression test for a real bug: all four adapters' *raw* S11 output
    (get_scattering_angle() for MSTM, the mueller.h5/result["mueller"]
    array for FaSTMM2) follows the standard Bohren-Huffman convention
    dCsca/dOmega = S11/k**2 -- i.e. integral(S11 dOmega) over the sphere
    equals k**2*Csca, confirmed empirically -- not the radiative-transfer
    phase-function convention (integral(S11 dOmega) == 4*pi) that
    ScatterResult.mueller's docstring promises and that a "phase function"
    is expected to obey by definition. Each adapter must rescale by
    4*pi/(k**2*Csca) before returning S11/S12.

    Uses a single sphere (azimuthally symmetric, so S11 depends only on
    theta) so integral(S11 dOmega) reduces to a clean 1D polar integral:
    2*pi * integral(S11(theta)*sin(theta) dtheta, theta=0..pi).
    """
    import numpy as np

    adapter = adapter_cls()
    if not adapter.is_available():
        pytest.skip(f"{adapter.name} not available")

    k = 2 * 3.141592653589793 / 0.5
    request = ClusterRequest(
        coords=[(0.0, 0.0, 0.0)],
        radii=[1.0],
        refractive_index=[(1.5, 0.01)],
        wavenumber=k,
        n_theta=181,
        n_phi=1,
        tolerance=1e-8,
        max_iterations=2000,
        formulation=0,
        compute_mueller=True,
    )
    result = adapter.solve(request)
    assert result.mueller is not None

    m = np.array(result.mueller)
    theta_rad = np.radians(m[:, 0])
    s11 = m[:, 1]
    integral = 2 * np.pi * np.trapezoid(s11 * np.sin(theta_rad), theta_rad)
    assert integral == pytest.approx(4 * 3.141592653589793, rel=0.01), (
        f"{adapter.name}: S11 does not integrate to 4*pi over the sphere "
        f"(got {integral:.4f}) -- not a properly normalized phase function"
    )
