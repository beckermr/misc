import os

import galsim
import piff


class DES_Piff(object):
    """A wrapper for Piff to use with Galsim.

    Parameters
    ----------
    file_name : str
        The file with the Piff psf solution.
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

    def getPSF(self, image_pos, wcs, x_interpolant='lanczos15', gsparams=None):
        """Get an image of the PSF at the given location.

        Parameters
        ----------
        image_pos : galsim.Position
            The image position for the PSF.
        wcs : galsim.BaseWCS or subclass
            The WCS to use to draw the PSF.
        x_interpolant : str, optional
            The interpolant to use.
        gsparams : galsim.GSParams, optional
            Ootional galsim configuration data to pass along.

        Returns
        -------
        psf : galsim.InterpolatedImage
            The PSF at the image position.
        """
        # nice and big image size here cause this has been a problem
        image = galsim.ImageD(ncol=53, nrow=53, wcs=wcs.local(image_pos))

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
