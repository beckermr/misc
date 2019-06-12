import numpy as np
import tqdm

import esutil as eu
import ngmix

from .metacal.metacal_fitter import MetacalFitter, METACAL_TYPES


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


def run_metacal(image, psf_image, stamp_size, noise, rng):
    """Run metacal on an image composed of stamps w/ constant noise.

    Parameters
    ----------
    image : np.ndarray
        The image of stamos.
    psf_image : np.ndarray
        The image of PSF images.
    stamp_size : int
        The size of each stamp.
    noise : float
        The noise level in the image.
    rng : np.random.RandomState
        An RNG to use.

    Returns
    -------
    result : dict
        A dictionary with each of the metacal catalogs.
    """
    nx = image.shape[1] // stamp_size
    ny = image.shape[0] // stamp_size
    cen = (stamp_size - 1) / 2

    def _gen_data():
        for yind in range(ny):
            for xind in range(nx):
                im = image[
                    yind*stamp_size:(yind+1)*stamp_size,
                    xind*stamp_size:(xind+1)*stamp_size]
                jac = ngmix.DiagonalJacobian(
                    scale=0.263,
                    x=cen,
                    y=cen)

                psf_im = psf_image[
                    yind*stamp_size:(yind+1)*stamp_size,
                    xind*stamp_size:(xind+1)*stamp_size]
                psf_noise = np.sqrt(np.sum(psf_im**2)) / 500
                wgt = 0.0 * im + 1.0 / psf_noise**2

                psf_obs = ngmix.Observation(
                    image=psf_im,
                    weight=wgt,
                    jacobian=jac
                )

                wgt = 0.0 * im + 1.0 / noise**2
                nse = rng.normal(size=im.shape) * noise
                obs = ngmix.Observation(
                    image=im,
                    weight=wgt,
                    noise=nse,
                    bmask=np.zeros_like(im, dtype=np.int32),
                    ormask=np.zeros_like(im, dtype=np.int32),
                    jacobian=jac,
                    psf=psf_obs
                )
                mbobs = ngmix.MultiBandObsList()
                obslist = ngmix.ObsList()
                obslist.append(obs)
                mbobs.append(obslist)

                mbobs.meta['id'] = xind + nx*yind
                # these settings do not matter that much I think
                mbobs[0].meta['Tsky'] = 1
                mbobs[0].meta['magzp_ref'] = 26.5
                mbobs[0][0].meta['orig_col'] = xind
                mbobs[0][0].meta['orig_row'] = yind

                yield mbobs

    data = []
    for mbobs in tqdm.tqdm(_gen_data(), total=nx*ny):
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
                continue
            elif sh_col in data.dtype.names:
                sh_cat[col] = data[sh_col]
            else:
                raise ValueError("column %s not found!" % col)
        result[sh] = sh_cat

    return result
