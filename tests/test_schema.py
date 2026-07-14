import pytest

from tbench.schema import ClusterRequest


def test_valid_request():
    req = ClusterRequest(
        coords=[(0.0, 0.0, 0.0)], radii=[1.0], refractive_index=[(1.5, 0.0)],
        wavenumber=1.0,
    )
    assert req.n_spheres == 1


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        ClusterRequest(
            coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            radii=[1.0],
            refractive_index=[(1.5, 0.0)],
            wavenumber=1.0,
        )


def test_defaults():
    req = ClusterRequest(
        coords=[(0.0, 0.0, 0.0)], radii=[1.0], refractive_index=[(1.5, 0.0)],
        wavenumber=1.0,
    )
    assert req.medium_refractive_index == (1.0, 0.0)
    assert req.n_theta == 181
    assert req.n_phi == 1
    assert req.extra == {}
