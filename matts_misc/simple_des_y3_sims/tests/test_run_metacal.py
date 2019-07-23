import numpy as np
import pytest
import ngmix
from ngmix import ObsList, Observation, MultiBandObsList

from ..run_metacal import _apply_pixel_scale, _strip_coadd, _strip_zero_flux


@pytest.fixture
def mbobs():
    mbobs = MultiBandObsList()
    for i in range(3):
        ol = ObsList()
        for j in range(4):
            o = Observation(
                image=np.ones((32, 32)) * (j + 1),
                weight=np.ones((32, 32)) * (j + 1),
                jacobian=ngmix.DiagonalJacobian(scale=0.25, row=0, col=0))
            ol.append(o)
        mbobs.append(ol)
    return mbobs


def test_apply_pixel_scale(mbobs):
    mbobs = _apply_pixel_scale(mbobs)
    for ol in mbobs:
        for j, o in enumerate(ol):
            assert np.allclose(o.image, (j+1) / 0.25**2)
            assert np.allclose(o.weight, (j+1) * 0.25**4)


def test_strip_coadd(mbobs):
    mbobs = _strip_coadd(mbobs)
    for ol in mbobs:
        for j, o in enumerate(ol):
            assert np.allclose(o.image, (j+2))
            assert np.allclose(o.weight, (j+2))


def test_strip_zero_flux(mbobs):
    mbobs[0][2].image = np.zeros((32, 32))
    mbobs[0][1].image = np.zeros((32, 32))

    mbobs[1][0].image = np.zeros((32, 32))

    mbobs[2][0].image = np.zeros((32, 32))
    mbobs[2][1].image = np.zeros((32, 32))
    mbobs[2][2].image = np.zeros((32, 32))

    mbobs = _strip_zero_flux(mbobs)

    assert len(mbobs[0]) == 2
    assert len(mbobs[1]) == 3
    assert len(mbobs[2]) == 1
