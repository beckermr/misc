import logging
import numpy as np
import os

import galsim
import galsim.config

import piff

LOGGER = logging.getLogger(__name__)


class DES_Piff(object):
    """A wrapper for Piff to use with Galsim.

    Parameters
    ----------
    file_name : str
        The file with the Piff psf solution.
    smooth : bool
        If True, use ngmix to fit a two-gaussian smooth model to the
        PSF images and return that model.
    """
    _req_params = {'file_name': str}
    _opt_params = {}
    _single_params = []
    _takes_rng = False

    def __init__(self, file_name):
        self.file_name = file_name
        self._piff = piff.read(
            os.path.expanduser(os.path.expandvars(file_name)))

    def getPiff(self):
        return self._piff

    def getPSF(self, image_pos, wcs, smooth=False,
               x_interpolant='lanczos15', gsparams=None):
        """Get an image of the PSF at the given location.

        Parameters
        ----------
        image_pos : galsim.Position
            The image position for the PSF.
        wcs : galsim.BaseWCS or subclass
            The WCS to use to draw the PSF.
        smooth : bool
            If True, return a smoothed version of the PSF.
        x_interpolant : str, optional
            The interpolant to use.
        gsparams : galsim.GSParams, optional
            Ootional galsim configuration data to pass along.

        Returns
        -------
        psf : galsim.GSObject
            The PSF at the image position.
        """
        if smooth:
            return self._make_smooth_psf_obj(
                image_pos, wcs, x_interpolant, gsparams)
        else:
            return self._make_psf_obj(image_pos, wcs, x_interpolant, gsparams)

    def _make_smooth_psf_obj(self, image_pos, wcs, x_interpolant, gsparams):
        import ngmix
        from ngmix.bootstrap import EMRunner

        swcs = galsim.PixelScale(0.263)

        # First we render the piff PSF with a simple pixel scale
        # piff offsets the center of the PSF from the true image
        # center - here we will return a properly centered image by undoing
        # the offset
        dx = image_pos.x - int(image_pos.x + 0.5)
        dy = image_pos.y - int(image_pos.y + 0.5)
        image = galsim.ImageD(ncol=33, nrow=33, wcs=swcs)
        psf = self.getPiff().draw(
            image_pos.x,
            image_pos.y,
            image=image,
            offset=(-dx, -dy))

        # now build the interpolated image
        psf = galsim.InterpolatedImage(
            psf,
            wcs=swcs,
            gsparams=gsparams,
            x_interpolant=x_interpolant
        ).withFlux(
            1.0
        )

        # remove the pixel
        psf = galsim.Convolve([
            galsim.Deconvolve(galsim.Pixel(0.263)),
            psf])
        im = psf.drawImage(nx=33, ny=33, wcs=swcs, method='no_pixel').array
        im /= np.sum(im)

        # now fit 2-gaussian model
        wgt = im * 0.0 + 1.0 / np.var(im[16-8, :])
        obs = ngmix.Observation(
            image=im,
            weight=wgt,
            jacobian=ngmix.Jacobian(x=16, y=16, wcs=swcs.jacobian())
        )

        ngauss = 2
        Tguess = 4.0
        ntry = 5
        em_pars = {'maxiter': 1000, 'tol': 1.0e-6}

        runner = EMRunner(obs, Tguess, ngauss, em_pars)
        runner.go(ntry=ntry)

        fitter = runner.get_fitter()
        res = fitter.get_result()
        assert res['flags'] == 0

        # finally return a galsim object
        return fitter.get_gmix().make_galsim_object()

    def _make_psf_obj(self, image_pos, wcs, x_interpolant, gsparams):
        image = galsim.ImageD(ncol=33, nrow=33, wcs=wcs.local(image_pos))

        # piff offsets the center of the PSF from the true image
        # center - here we will return a properly centered image by undoing
        # the offset
        dx = image_pos.x - int(image_pos.x + 0.5)
        dy = image_pos.y - int(image_pos.y + 0.5)

        psf = self.getPiff().draw(
            image_pos.x,
            image_pos.y,
            image=image,
            offset=(-dx, -dy))

        psf = galsim.InterpolatedImage(
            psf,
            wcs=wcs.local(image_pos),
            gsparams=gsparams,
            x_interpolant=x_interpolant
        ).withFlux(
            1.0
        )

        return psf


class PiffLoader(galsim.config.InputLoader):
    def getKwargs(self, config, base, logger):
        req = {'file_name': str}
        opt = {}
        kwargs, safe = galsim.config.GetAllParams(
            config, base, req=req, opt=opt)

        return kwargs, safe


# add a config input section
galsim.config.RegisterInputType('des_piff', PiffLoader(DES_Piff))


# and a builder
def BuildDES_Piff(config, base, ignore, gsparams, logger):
    des_piff = galsim.config.GetInputObj('des_piff', config, base, 'DES_Piff')

    opt = {'flux': float,
           'num': int,
           'image_pos': galsim.PositionD,
           'smooth': bool,
           'x_interpolant': str}
    params, safe = galsim.config.GetAllParams(
        config, base, opt=opt, ignore=ignore)

    if 'image_pos' in params:
        image_pos = params['image_pos']
    elif 'image_pos' in base:
        image_pos = base['image_pos']
    else:
        raise galsim.GalSimConfigError(
            "DES_Piff requested, but no image_pos defined in base.")

    if 'wcs' not in base:
        raise galsim.GalSimConfigError(
            "DES_Piff requested, but no wcs defined in base.")
    wcs = base['wcs']

    if gsparams:
        gsparams = galsim.GSParams(**gsparams)
    else:
        gsparams = None

    psf = des_piff.getPSF(
        image_pos,
        wcs,
        gsparams=gsparams,
        smooth=params.get('smooth', False),
        x_interpolant=params.get('x_interpolant', 'lanczos15'))

    if 'flux' in params:
        psf = psf.withFlux(params['flux'])

    # we make sure to declare the returned object as not safe for reuse
    can_be_reused = False
    return psf, can_be_reused


galsim.config.RegisterObjectType(
    'DES_Piff', BuildDES_Piff, input_type='des_piff')
