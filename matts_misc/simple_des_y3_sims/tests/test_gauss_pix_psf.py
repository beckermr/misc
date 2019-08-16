import numpy as np
import galsim

from ..gauss_pix_psf import GaussPixPSF


def test_gauss_pix_psf_smoke():
    psf_model = GaussPixPSF()
    psf = psf_model.getPSF(galsim.PositionD(x=1, y=2))
    psf_im = psf.drawImage(nx=53, ny=53, scale=0.263, method='no_pixel').array

    seed = 4098
    rng = np.random.RandomState(seed=seed)
    g1 = rng.normal() * 0.01
    g2 = rng.normal() * 0.01
    fwhm = (rng.uniform(low=-0.1, high=0.1) + 1.0) * 0.9
    gs = galsim.Gaussian(fwhm=fwhm).shear(g1=g1, g2=g2).withFlux(1.0)
    test_im = gs.drawImage(nx=53, ny=53, scale=0.263).array

    assert np.allclose(psf_im, test_im, atol=1e-5, rtol=0)


def test_gauss_pix_psf_reproducible():
    psf_model = GaussPixPSF()
    psf = psf_model.getPSF(galsim.PositionD(x=1, y=2))
    psf_im1 = psf.drawImage(nx=53, ny=53, scale=0.263, method='no_pixel').array

    psf_model = GaussPixPSF()
    psf = psf_model.getPSF(galsim.PositionD(x=1, y=2))
    psf_im2 = psf.drawImage(nx=53, ny=53, scale=0.263, method='no_pixel').array

    assert np.allclose(psf_im1, psf_im2)


def test_gauss_pix_psf_s2n():
    psf_model = GaussPixPSF(s2n=100)
    psf = psf_model.getPSF(galsim.PositionD(x=1, y=2))
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
