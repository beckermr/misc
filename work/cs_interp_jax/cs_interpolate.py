import numpy as np
import scipy.fftpack as spfft

from lbfgs import fmin_lbfgs


def _dct2(x):
    "A 2d discrete cosine transform."
    return spfft.dct(spfft.dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')


def _idct2(x):
    "An inverse 2d discrete cosine transform."
    return spfft.idct(
        spfft.idct(x, axis=0, norm='ortho'), axis=1, norm='ortho')


def _interpolate_image_cs(image, sample_mask, c=15):
    ri_vector, = np.where(sample_mask.ravel())
    b_vector = image.ravel()[ri_vector].copy()
    image_dims = image.shape

    def _evaluate(x, g):
        # return squared norm of residuals and set the gradient
        x2 = x.reshape(image_dims)
        Ax2 = _idct2(x2)
        Ax = Ax2.flat[ri_vector]
        Axb = Ax - b_vector
        fx = np.sum(np.power(Axb, 2))
        Axb2 = np.zeros(x2.shape)
        Axb2.flat[ri_vector] = Axb
        AtAxb2 = 2 * _dct2(Axb2)
        AtAxb = AtAxb2.reshape(x.shape)

        np.copyto(g, AtAxb)
        return fx

    x0 = np.zeros_like(image).ravel()
    x = fmin_lbfgs(_evaluate, x0, orthantwise_c=c, line_search='wolfe')

    # transform the output back into the spatial domain
    x = _idct2(x.reshape(image_dims))

    return x


def interpolate_image_and_noise_cs(
        *, image, noise, weight, bmask, bad_flags, rng,
        sampling_rate=1, c=100):
    """Interpolate an image and a noise field using compressed sensing in the
    Fourier domain.

    Parameters
    ----------
    image : array-like
        The image to interpolate.
    noise : array-like
        The noise field to interpolate.
    weight : array-like
        A weight map to test for zero values. Any pixels with zero weight
        are interpolated.
    bmask : array-like
        The bit mask for the slice.
    bad_flags : int
        Pixels with in the bit mask using
        `(bmask & bad_flags) != 0`.
    rng : `numpy.random.RandomState`
        An RNG instance to use.
    sampling_rate : float
        The rate at which to sample the good pixels for the solver.
    c : float
        The weight of the L1 penalty in the compressed sensing loss.

    Returns
    -------
    interp_image : array-like
        The interpolated image.
    interp_noise : array-like
        The interpolated noise image.
    """
    bad_mask = (weight <= 0) | ((bmask & bad_flags) != 0)

    if np.any(bad_mask):
        # get the pixels to sample
        good_mask = ~bad_mask
        good_inds = np.where(good_mask)
        assert np.all(good_mask[good_inds[0], good_inds[1]])

        n_sample = int(np.ceil(len(good_inds[0]) * sampling_rate))
        assert n_sample > 0

        sample_mask = np.zeros(image.shape)
        inds = rng.choice(len(good_inds[0]), replace=False, size=n_sample)
        sample_mask[good_inds[0][inds], good_inds[1][inds]] = 1
        sample_mask = sample_mask.astype(np.bool)

        # interpolate
        interp_image = _interpolate_image_cs(image, sample_mask, c=c)
        interp_image[good_mask] = image[good_mask]

        interp_noise = _interpolate_image_cs(noise, sample_mask, c=c)
        interp_noise[good_mask] = noise[good_mask]

        return interp_image, interp_noise
    else:
        # return a copy here since the caller expects new images
        return image.copy(), noise.copy()
