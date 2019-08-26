import sys
import numpy as np
import tqdm

import ngmix
import galsim
import numba

from ngmix.fitting import LMSimple


def run_ml(*, n_sims, wcs_g1, wcs_g2):
    """Run the moments fitter and check g1, g2.

    The resulting values are printed to STDOUT.

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

    res = _run_ml(
        n_sims=n_sims,
        rng=np.random.RandomState(seed=10),
        **jacobian_dict)

    g1 = np.array([r['g'][0] for r in res])
    g2 = np.array([r['g'][1] for r in res])
    g1m, g1_err, g2m, g2_err = _jack_est(g1, g2)

    print("""\
# of sims: {n_sims}
wcs_g1   : {wcs_g1:f}
wcs_g2   : {wcs_g2:f}
dudx     : {dudx:f}
dudy     : {dudy:f}
dvdx     : {dvdx:f}
dvdy     : {dvdy:f}
g1 [1e-3] : {g1m:f} +/- {g1_err:f}
g2 [1e-3] : {g2m:f} +/- {g2_err:f}""".format(
        n_sims=len(g1),
        wcs_g1=wcs_g1,
        wcs_g2=wcs_g2,
        **jacobian_dict,
        g1m=g1m/1e-3,
        g1_err=g1_err/1e-3,
        g2m=g2m/1e-3,
        g2_err=g2_err/1e-3), flush=True)


@numba.njit
def _jack_est(g1, g2):
    g1bar = np.mean(g1)
    g2bar = np.mean(g2)
    n = g1.shape[0]
    fac = n / (n-1)
    g1_samps = np.zeros_like(g1)
    g2_samps = np.zeros_like(g2)

    for i in range(n):
        _g1 = fac * (g1bar - g1[i]/n)
        _g2 = fac * (g2bar - g2[i]/n)
        g1_samps[i] = _g1
        g2_samps[i] = _g2

    g1m = np.mean(g1_samps)
    g2m = np.mean(g2_samps)

    g1_err = np.sqrt(np.sum((g1m - g1_samps)**2) / fac)
    g2_err = np.sqrt(np.sum((g2m - g2_samps)**2) / fac)

    return g1m, g1_err, g2m, g2_err


def _run_ml(*, n_sims, rng, dudx, dudy, dvdx, dvdy):
    """Run metacal on an image composed of stamps w/ constant noise.

    Parameters
    ----------
    n_sims : int
        The number of objects to run.
    rng : np.random.RandomState
        An RNG to use.
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

    stamp_size = 33
    psf_stamp_size = 33

    cen = (stamp_size - 1) / 2
    psf_cen = (psf_stamp_size - 1)/2

    s2n = 1e6
    flux = 1e6

    galsim_jac = galsim.JacobianWCS(
        dudx=dudx,
        dudy=dudy,
        dvdx=dvdx,
        dvdy=dvdy)

    gal = galsim.Exponential(
        half_light_radius=0.5
    ).withFlux(
        flux)
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
            wcs=galsim_jac).array
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
        offset = rng.uniform(low=-0.5, high=0.5, size=2)
        im = obj.drawImage(
            nx=stamp_size,
            ny=stamp_size,
            wcs=galsim_jac,
            offset=offset).array
        jac = ngmix.Jacobian(
            x=cen+offset[0],
            y=cen+offset[1],
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
            res = _run_ml_fitter(mbobs, rng)
        except Exception as e:
            print('err:', e, type(e))
            res = None

        if res is not None:
            data.append(res)

    if len(data) > 0:
        res = data
    else:
        res = None

    return res


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


def _run_ml_fitter(mbobs, rng):
    # fit the PSF
    _fit_psf(mbobs[0][0].psf)

    # overall flags, or'ed from each moments fit
    res = {'mcal_flags': 0}
    try:
        fitter = LMSimple(mbobs[0][0], 'gauss')
        fitter.go(np.ones(6) * 0.1)
        fres = fitter.get_result()
    except Exception as err:
        print('err:', err)
        fres = {'flags': np.ones(1, dtype=[('flags', 'i4')])}

    res['mcal_flags'] |= fres['flags']
    if not res['mcal_flags']:
        res['g'] = fres['g']
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

    run_ml(n_sims=int(sys.argv[1]), wcs_g1=wcs_g1, wcs_g2=wcs_g2)
