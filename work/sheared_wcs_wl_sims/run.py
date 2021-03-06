import sys
import numpy as np
import tqdm

import ngmix
import galsim
import numba

from ngmix import metacal
from metadetect.fitting import Moments


def run_metacal(*, n_sims, wcs_g1, wcs_g2):
    """Run metacal and measure m and c.

    The resulting m and c are printed to STDOUT.

    Parameters
    ----------
    n_sims : int
        The number of objects to simulated.
    wcs_g1 : float
        The shear on the 1-axis of the WCS Jacobian.
    wcs_g2 : float
        The shear on the 2-axis of the WCS Jacobian.
    """
    jc = galsim.ShearWCS(0.263, galsim.Shear(g1=wcs_g1, g2=wcs_g2)).jacobian()

    jacobian_dict = {
        'dudx': jc.dudx,
        'dudy': jc.dudy,
        'dvdx': jc.dvdx,
        'dvdy': jc.dvdy
    }

    swap_g1g2 = False

    res = _run_metacal(
        n_sims=n_sims,
        rng=np.random.RandomState(seed=10),
        swap_g1g2=swap_g1g2,
        **jacobian_dict)

    g1 = np.array([r['noshear']['g'][0] for r in res])
    g2 = np.array([r['noshear']['g'][1] for r in res])
    g1p = np.array([r['1p']['g'][0] for r in res])
    g1m = np.array([r['1m']['g'][0] for r in res])
    g2p = np.array([r['2p']['g'][1] for r in res])
    g2m = np.array([r['2m']['g'][1] for r in res])

    g_true = 0.02
    step = 0.01

    if swap_g1g2:
        R11 = (g1p - g1m) / 2 / step
        R22 = (g2p - g2m) / 2 / step * g_true

        m, merr, c, cerr = _jack_est(g2, R22, g1, R11)
    else:
        R11 = (g1p - g1m) / 2 / step * g_true
        R22 = (g2p - g2m) / 2 / step

        m, merr, c, cerr = _jack_est(g1, R11, g2, R22)

    print("""\
# of sims: {n_sims}
wcs_g1   : {wcs_g1:f}
wcs_g2   : {wcs_g2:f}
dudx     : {dudx:f}
dudy     : {dudy:f}
dvdx     : {dvdx:f}
dvdy     : {dvdy:f}
m [1e-3] : {m:f} +/- {msd:f}
c [1e-4] : {c:f} +/- {csd:f}""".format(
        n_sims=len(g1),
        wcs_g1=wcs_g1,
        wcs_g2=wcs_g2,
        **jacobian_dict,
        m=m/1e-3,
        msd=merr/1e-3,
        c=c/1e-4,
        csd=cerr/1e-4), flush=True)


def _run_metacal(*, n_sims, rng, swap_g1g2, dudx, dudy, dvdx, dvdy):
    """Run metacal on an image composed of stamps w/ constant noise.

    Parameters
    ----------
    n_sims : int
        The number of objects to run.
    rng : np.random.RandomState
        An RNG to use.
    swap_g1g2 : bool
        If True, set the true shear on the 2-axis to 0.02 and 1-axis to 0.0.
        Otherwise, the true shear on the 1-axis is 0.02 and on the 2-axis is
        0.0.
    dudx : float
        The du/dx Jacobian component.
    dudy : float
        The du/dy Jacobian component.
    dydx : float
        The dv/dx Jacobian component.
    dvdy : float
        The dv/dy Jacobian component.

    Returns
    -------
    result : dict
        A dictionary with each of the metacal catalogs.
    """

    method = 'auto'

    stamp_size = 33
    psf_stamp_size = 33

    cen = (stamp_size - 1) / 2
    psf_cen = (psf_stamp_size - 1)/2

    s2n = 1e16
    flux = 1e6

    galsim_jac = galsim.JacobianWCS(
        dudx=dudx,
        dudy=dudy,
        dvdx=dvdx,
        dvdy=dvdy)

    if swap_g1g2:
        g1 = 0.0
        g2 = 0.02
    else:
        g1 = 0.02
        g2 = 0.0

    gal = galsim.Exponential(
        half_light_radius=0.5
    ).withFlux(
        flux
    ).shear(
        g1=g1, g2=g2)

    psf = galsim.Gaussian(fwhm=0.9).withFlux(1)
    obj = galsim.Convolve(gal, psf)
    obj_im = obj.drawImage(nx=111, ny=111).array
    noise = np.sqrt(np.sum(obj_im**2))/s2n

    data = []
    for ind in tqdm.trange(n_sims):
        ################################
        # make the obs

        # psf
        psf_im = psf.drawImage(
            nx=psf_stamp_size,
            ny=psf_stamp_size,
            wcs=galsim_jac,
            method=method).array
        psf_noise = np.sqrt(np.sum(psf_im**2)) / 10000
        wgt = np.ones_like(psf_im) / psf_noise**2
        psf_im += (rng.normal(size=psf_im.shape) * psf_noise)
        psf_jac = ngmix.Jacobian(
            x=psf_cen,
            y=psf_cen,
            dudx=dudx,
            dudy=dudy,
            dvdx=dvdx,
            dvdy=dvdy)
        psf_obs = ngmix.Observation(
            image=psf_im,
            weight=wgt,
            jacobian=psf_jac)

        # now render object
        scale = psf_jac.scale
        shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
        _obj = obj.shift(dx=shift[0], dy=shift[1])
        xy = galsim_jac.toImage(galsim.PositionD(shift))
        im = _obj.drawImage(
            nx=stamp_size,
            ny=stamp_size,
            wcs=galsim_jac,
            method=method).array
        jac = ngmix.Jacobian(
            x=cen+xy.x,
            y=cen+xy.y,
            dudx=dudx,
            dudy=dudy,
            dvdx=dvdx,
            dvdy=dvdy)
        wgt = np.ones_like(im) / noise**2
        nse = rng.normal(size=im.shape) * noise
        im += (rng.normal(size=im.shape) * noise)
        obs = ngmix.Observation(
            image=im,
            weight=wgt,
            noise=nse,
            bmask=np.zeros_like(im, dtype=np.int32),
            ormask=np.zeros_like(im, dtype=np.int32),
            jacobian=jac,
            psf=psf_obs
        )

        # build the mbobs
        mbobs = ngmix.MultiBandObsList()
        obslist = ngmix.ObsList()
        obslist.append(obs)
        mbobs.append(obslist)

        mbobs.meta['id'] = ind+1
        # these settings do not matter that much I think
        mbobs[0].meta['Tsky'] = 1
        mbobs[0].meta['magzp_ref'] = 26.5
        mbobs[0][0].meta['orig_col'] = ind+1
        mbobs[0][0].meta['orig_row'] = ind+1

        ################################
        # run the fitters
        try:
            res = _run_metacal_fitter(mbobs, rng)
        except Exception as e:
            print(e)
            res = None

        if res is not None:
            data.append(res)

    if len(data) > 0:
        res = data
    else:
        res = None

    return res


@numba.njit
def _jack_est(g1, R11, g2, R22):
    g1bar = np.mean(g1)
    R11bar = np.mean(R11)
    g2bar = np.mean(g2)
    R22bar = np.mean(R22)
    n = g1.shape[0]
    fac = n / (n-1)
    m_samps = np.zeros_like(g1)
    c_samps = np.zeros_like(g1)

    for i in range(n):
        _g1 = fac * (g1bar - g1[i]/n)
        _R11 = fac * (R11bar - R11[i]/n)
        _g2 = fac * (g2bar - g2[i]/n)
        _R22 = fac * (R22bar - R22[i]/n)
        m_samps[i] = _g1 / _R11 - 1
        c_samps[i] = _g2 / _R22

    m = np.mean(m_samps)
    c = np.mean(c_samps)

    m_err = np.sqrt(np.sum((m - m_samps)**2) / fac)
    c_err = np.sqrt(np.sum((c - c_samps)**2) / fac)

    return m, m_err, c, c_err


def _fit_psf(psf):
    runner = ngmix.bootstrap.PSFRunner(
        psf,
        'gauss',
        1.0,
        {'maxfev': 2000, 'ftol': 1.0e-5, 'xtol': 1.0e-5}
    )
    runner.go(ntry=2)

    psf_fitter = runner.fitter
    res = psf_fitter.get_result()
    psf.update_meta_data({'fitter': psf_fitter})

    if res['flags'] == 0:
        gmix = psf_fitter.get_gmix()
        psf.set_gmix(gmix)
    else:
        from ngmix.gexceptions import BootPSFFailure
        raise BootPSFFailure("failed to fit psfs: %s" % str(res))


def _run_metacal_fitter(mbobs, rng):
    # fit the PSF
    _fit_psf(mbobs[0][0].psf)

    metacal_pars = {
        'psf': 'fitgauss',
        'types': ['noshear', '1p', '1m', '2p', '2m'],
        'use_noise_image': True,
        'step': 0.01
    }
    moments_pars = {'bmask_flags': 2**30, 'weight': {'fwhm': 1.2}}

    obs_dict = metacal.get_all_metacal(mbobs, **metacal_pars)

    # overall flags, or'ed from each moments fit
    res = {'mcal_flags': 0}
    for key in sorted(obs_dict):
        try:
            fitter = Moments(moments_pars, rng)
            fres = fitter.go([obs_dict[key]])
        except Exception as err:
            print(err)
            fres = {'flags': np.ones(1, dtype=[('flags', 'i4')])}

        res['mcal_flags'] |= fres['flags'][0]
        tres = {}
        for name in fres.dtype.names:
            no_wmom = name.replace('wmom_', '')
            tres[no_wmom] = fres[name][0]
        tres['flags'] = fres['flags'][0]  # make sure this is moved over
        res[key] = tres

    return res


if __name__ == '__main__':
    if len(sys.argv) > 2:
        wcs_g1 = float(sys.argv[2])
    else:
        wcs_g1 = 0.0

    if len(sys.argv) > 3:
        wcs_g2 = float(sys.argv[3])
    else:
        wcs_g2 = wcs_g1

    run_metacal(n_sims=int(sys.argv[1]), wcs_g1=wcs_g1, wcs_g2=wcs_g2)
