import numpy as np
import tqdm

import esutil as eu
import ngmix
import galsim

from ..des_piff import DES_Piff
from ..metacal.metacal_fitter import MetacalFitter, METACAL_TYPES


CONFIG = {
    'metacal': {
        # check for an edge hit
        'bmask_flags': 2**30,

        'metacal_pars': {
            'psf': 'fitgauss',
            'types': ['noshear', '1p', '1m', '2p', '2m'],
            'use_noise_image': True,
        },

        'model': 'gauss',

        'max_pars': {
            'ntry': 2,
            'pars': {
                'method': 'lm',
                'lm_pars': {
                    'maxfev': 2000,
                    'xtol': 5.0e-5,
                    'ftol': 5.0e-5,
                }
            }
        },

        'priors': {
            'cen': {
                'type': 'normal2d',
                'sigma': 0.263
            },

            'g': {
                'type': 'ba',
                'sigma': 0.2
            },

            'T': {
                'type': 'two-sided-erf',
                'pars': [-1.0, 0.1, 1.0e+06, 1.0e+05]
            },

            'flux': {
                'type': 'two-sided-erf',
                'pars': [-100.0, 1.0, 1.0e+09, 1.0e+08]
            }
        },

        'psf': {
            'model': 'gauss',
            'ntry': 2,
            'lm_pars': {
                'maxfev': 2000,
                'ftol': 1.0e-5,
                'xtol': 1.0e-5
            }
        }
    },
}


def run_metacal(n_sims, stamp_size, psf_stamp_size, rng,
                jacobian_dict, gauss_psf):
    """Run metacal on an image composed of stamps w/ constant noise.

    Parameters
    ----------
    n_sims : int
        The number of objects to run.
    stamp_size : int
        The size of each stamp.
    psf_stamp_size : int
        The size of each PSF stamp.
    rng : np.random.RandomState
        An RNG to use.
    jacobian_dict : dict
        A dictonary with the components of the image jacobian.
    gauss_psf : bool
        If True, test with a Gaussian PSF.

    Returns
    -------
    result : dict
        A dictionary with each of the metacal catalogs.
    """

    method = 'auto'

    cen = (stamp_size - 1) / 2
    psf_cen = (psf_stamp_size - 1)/2
    noise = 1
    flux = 64000
    gal = galsim.Exponential(
        half_light_radius=0.5
    ).withFlux(
        flux
    ).shear(
        g1=0.02, g2=0.0)

    if not gauss_psf:
        piff_cats = np.loadtxt('piff_cat.txt', dtype=str)
        piff_file = rng.choice(piff_cats)
        psf_model = DES_Piff(piff_file)
        print('piff file:', piff_file)

    if jacobian_dict['dudy'] != 0 or jacobian_dict['dvdx'] != 0:
        print('jacobian:', jacobian_dict)

    galsim_jac = galsim.JacobianWCS(
        dudx=jacobian_dict['dudx'],
        dudy=jacobian_dict['dudy'],
        dvdx=jacobian_dict['dvdx'],
        dvdy=jacobian_dict['dvdy'])

    data = []
    for ind in tqdm.trange(n_sims):
        ################################
        # make the obs

        # first get PSF
        x = rng.uniform(low=1, high=2048)
        y = rng.uniform(low=1, high=2048)

        if gauss_psf:
            psf = galsim.Gaussian(fwhm=0.9).withFlux(1)
        else:
            psf = psf_model.getPSF(
                galsim.PositionD(x=x, y=y),
                galsim_jac)

        psf_im = psf.drawImage(
            nx=psf_stamp_size,
            ny=psf_stamp_size,
            wcs=galsim_jac,
            method=method).array

        psf_noise = np.sqrt(np.sum(psf_im**2)) / 500
        wgt = 0.0 * psf_im + 1.0 / psf_noise**2
        psf_jac = ngmix.Jacobian(
            x=psf_cen,
            y=psf_cen,
            dudx=jacobian_dict['dudx'],
            dudy=jacobian_dict['dudy'],
            dvdx=jacobian_dict['dvdx'],
            dvdy=jacobian_dict['dvdy'])
        psf_obs = ngmix.Observation(
            image=psf_im,
            weight=wgt,
            jacobian=psf_jac
        )

        # now render object
        obj = galsim.Convolve(gal, psf)
        offset = rng.uniform(low=-0.5, high=0.5, size=2)
        im = obj.drawImage(
            nx=stamp_size,
            ny=stamp_size,
            wcs=galsim_jac,
            offset=offset,
            method=method).array
        jac = ngmix.Jacobian(
            x=cen+offset[0],
            y=cen+offset[1],
            dudx=jacobian_dict['dudx'],
            dudy=jacobian_dict['dudy'],
            dvdx=jacobian_dict['dvdx'],
            dvdy=jacobian_dict['dvdy'])
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
        # run the fitter
        mcal = MetacalFitter(CONFIG, 1, rng)
        mcal.go([mbobs])
        res = mcal.result

        if res is not None:
            data.append(res)

    if len(data) > 0:
        res = eu.numpy_util.combine_arrlist(data)
        result = _result_to_dict(res)
    else:
        result = None

    return result


def _result_to_dict(data):
    cols_to_always_keep = ['x', 'y']

    def _get_col_type(col):
        for dtup in data.descr.descr:
            if dtup[0] == col:
                return list(dtup[1:])
        return None

    result = {}

    # now build each of other catalogs
    for sh in METACAL_TYPES:
        dtype_descr = []
        for dtup in data.dtype.descr:
            if dtup[0] in cols_to_always_keep:
                dtype_descr.append(dtup)
            elif dtup[0].startswith('mcal_') and dtup[0].endswith(sh):
                dlist = [dtup[0].replace('_%s' % sh, '')]
                dlist = dlist + list(dtup[1:])
                dtype_descr.append(tuple(dlist))

        sh_cat = np.zeros(len(data), dtype=dtype_descr)
        for col in sh_cat.dtype.names:
            sh_col = col + '_%s' % sh
            if col in data.dtype.names:
                sh_cat[col] = data[col]
            elif sh_col in data.dtype.names:
                sh_cat[col] = data[sh_col]
            else:
                raise ValueError("column %s not found!" % col)
        result[sh] = sh_cat

    return result
