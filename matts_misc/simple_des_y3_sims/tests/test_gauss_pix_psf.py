import numpy as np
import galsim
import pytest

from ..gauss_pix_psf import GaussPixPSF


@pytest.mark.parametrize('wcs', [
    galsim.PixelScale(0.263),
    galsim.PixelScale(0.5),
    galsim.JacobianWCS(-0.2634420129421214, 0.0006503502840044033,
                       -0.0003966040125006026, -0.26354105026622404)])
def test_gauss_pix_psf_smoke(wcs):
    psf_model = GaussPixPSF()
    psf = psf_model.getPSF(galsim.PositionD(x=1, y=2), wcs)
    psf_im = psf.drawImage(nx=53, ny=53, wcs=wcs, method='no_pixel').array

    seed = 4098
    rng = np.random.RandomState(seed=seed)
    g1 = rng.normal() * 0.01
    g2 = rng.normal() * 0.01
    fwhm = (rng.uniform(low=-0.1, high=0.1) + 1.0) * 0.9
    gs = galsim.Gaussian(fwhm=fwhm).shear(g1=g1, g2=g2).withFlux(1.0)
    test_im = gs.drawImage(nx=53, ny=53, wcs=wcs).array

    assert np.allclose(psf_im, test_im, atol=1e-5, rtol=0)


def test_gauss_pix_psf_reproducible():
    wcs = galsim.PixelScale(0.5)
    psf_model = GaussPixPSF()
    psf = psf_model.getPSF(galsim.PositionD(x=1, y=2), wcs)
    psf_im1 = psf.drawImage(nx=53, ny=53, scale=0.263, method='no_pixel').array

    psf_model = GaussPixPSF()
    psf = psf_model.getPSF(galsim.PositionD(x=1, y=2), wcs)
    psf_im2 = psf.drawImage(nx=53, ny=53, scale=0.263, method='no_pixel').array

    assert np.allclose(psf_im1, psf_im2)


def test_gauss_pix_psf_s2n():
    wcs = galsim.PixelScale(0.263)
    psf_model = GaussPixPSF(s2n=100)
    psf = psf_model.getPSF(galsim.PositionD(x=1, y=2), wcs)
    psf_im = psf.drawImage(nx=53, ny=53, scale=0.263, method='no_pixel').array

    seed = 4098
    rng = np.random.RandomState(seed=seed)
    g1 = rng.normal() * 0.01
    g2 = rng.normal() * 0.01
    fwhm = (rng.uniform(low=-0.1, high=0.1) + 1.0) * 0.9
    gs = galsim.Gaussian(fwhm=fwhm).shear(g1=g1, g2=g2).withFlux(1.0)
    test_im = gs.drawImage(nx=53, ny=53, scale=0.263).array

    noise_std = np.sqrt(np.sum(test_im**2)/100**2)
    assert np.allclose(np.std(psf_im - test_im), noise_std, atol=1e-4, rtol=0)
