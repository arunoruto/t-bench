import numpy as np
import pytest

from tbench.sweep import MaterialSpec, SweepRequest, expand_sweep

_TWO_SPHERES = dict(
    coords=[(-1.5, 0.0, 0.0), (1.5, 0.0, 0.0)],
    radii=[1.0, 1.0],
)


def test_material_spec_requires_exactly_one_source():
    with pytest.raises(ValueError):
        MaterialSpec()
    with pytest.raises(ValueError):
        MaterialSpec(
            refractive_index=(1.5, 0.0),
            refidxdb_url="https://refractiveindex.info/x",
        )


def test_material_spec_fixed_refractive_index_broadcasts():
    m = MaterialSpec(refractive_index=(1.5, 0.01))
    nk = m.refractive_index_at(np.array([0.5, 0.7, 1.0]))
    assert nk.shape == (3,)
    assert np.all(nk == complex(1.5, 0.01))


def test_expand_sweep_one_request_per_wavelength():
    sweep = SweepRequest(
        **_TWO_SPHERES,
        material=MaterialSpec(refractive_index=(1.5, 0.01)),
        wavelengths_um=[0.5, 0.7, 1.0],
    )
    requests = expand_sweep(sweep)
    assert len(requests) == 3
    for req, wl in zip(requests, sweep.wavelengths_um):
        assert req.n_spheres == 2
        assert req.refractive_index == [(1.5, 0.01), (1.5, 0.01)]
        assert req.wavenumber == pytest.approx(2 * np.pi / wl)
        assert req.coords == [(-1.5, 0.0, 0.0), (1.5, 0.0, 0.0)]
        assert req.radii == [1.0, 1.0]


def test_sweep_requires_matching_coords_and_radii_lengths():
    with pytest.raises(ValueError):
        SweepRequest(
            coords=[(0.0, 0.0, 0.0)],
            radii=[1.0, 2.0],
            material=MaterialSpec(refractive_index=(1.5, 0.0)),
            wavelengths_um=[0.5],
        )


def test_sweep_requires_nonempty_wavelengths():
    with pytest.raises(ValueError):
        SweepRequest(
            **_TWO_SPHERES,
            material=MaterialSpec(refractive_index=(1.5, 0.0)),
            wavelengths_um=[],
        )


def test_material_spec_refidxdb_url_real_data():
    """Cross-checks against refidxdb's own cached refractiveindex.info
    data (SiO2) -- skipped if the cache/network isn't available."""
    m = MaterialSpec(
        refidxdb_url=(
            "https://refractiveindex.info/database/data/main/SiO2/"
            "nk/Rodriguez-de%20Marcos.yml"
        )
    )
    try:
        nk = m.refractive_index_at(np.array([0.5, 1.0]))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"refidxdb data not available: {exc}")

    # Known fused-silica refractive index in the visible/near-IR.
    assert nk[0].real == pytest.approx(1.468, abs=0.01)
    assert nk[1].real == pytest.approx(1.459, abs=0.01)
