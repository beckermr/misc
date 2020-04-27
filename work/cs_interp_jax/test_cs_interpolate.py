import time

import numpy as np
import galsim
import ngmix

from cs_interpolate import interpolate_image_and_noise_cs


def test_interpolate_image_and_noise_cs_gauss_linear(show=False):
    """
    test that our interpolation works decently for a linear
    piece missing from a gaussian image
    """

    rng = np.random.RandomState(seed=31415)
    noise = 0.001

    sigma = 4.0
    is2 = 1.0/sigma**2
    dims = 51, 51
    cen = (np.array(dims)-1.0)/2.0

    rows, cols = np.mgrid[
        0:dims[0],
        0:dims[1],
    ]
    rows = rows - cen[0]
    cols = cols - cen[1]

    image_unmasked = np.exp(-0.5*(rows**2 + cols**2)*is2)
    weight = image_unmasked*0 + 1.0/noise**2

    noise_image = rng.normal(scale=noise, size=image_unmasked.shape)

    badcol = int(cen[1]-3)
    bw = 3
    rr = badcol-bw, badcol+bw+1

    weight[rr[0]:rr[1], badcol] = 0.0
    image_masked = image_unmasked.copy()
    image_masked[rr[0]:rr[1], badcol] = 0.0

    bmask = np.zeros_like(image_unmasked, dtype=np.int32)
    bad_flags = 0

    iimage, inoise = interpolate_image_and_noise_cs(
        image=image_masked,
        weight=weight,
        bmask=bmask,
        bad_flags=bad_flags,
        noise=noise_image,
        rng=np.random.RandomState(seed=45),
        c=1,
    )

    maxdiff = np.abs(image_unmasked-iimage).max()

    if show:
        from espy import images
        images.view_mosaic([image_masked, weight])

        images.compare_images(
            image_unmasked,
            iimage,
            width=1000,
            height=int(1000*2/3),
        )
        print('max diff:', maxdiff)

    assert maxdiff < 0.5


def test_interpolate_image_and_noise_cs_gauss_circle(show=False):
    """
    test that our interpolation works decently for a linear
    piece missing from a gaussian image
    """

    rng = np.random.RandomState(seed=31415)
    noise = 0.001

    sigma = 4.0
    is2 = 1.0/sigma**2
    dims = 51, 51
    cen = (np.array(dims)-1.0)/2.0

    rows, cols = np.mgrid[
        0:dims[0],
        0:dims[1],
    ]
    rows = rows - cen[0]
    cols = cols - cen[1]
    radius2 = rows**2 + cols**2

    image_unmasked = np.exp(-0.5*(radius2)*is2)
    weight = image_unmasked*0 + 1.0/noise**2

    noise_image = rng.normal(scale=noise, size=image_unmasked.shape)

    wbad = np.where(radius2 <= 3**2)

    weight[wbad] = 0.0
    image_masked = image_unmasked.copy()
    image_masked[wbad] = 0.0

    bmask = np.zeros_like(image_unmasked, dtype=np.int32)
    bad_flags = 0

    iimage, inoise = interpolate_image_and_noise_cs(
        image=image_masked,
        weight=weight,
        bmask=bmask,
        bad_flags=bad_flags,
        noise=noise_image,
        rng=np.random.RandomState(seed=45),
        # sampling_rate=0.1,
        c=0.001,
    )

    # iimage[wbad] *= 2.75
    # iimage[wbad] *= 1.5

    maxdiff = np.abs(image_unmasked-iimage).max()

    if show:
        from espy import images
        images.view_mosaic([image_masked, weight])

        images.compare_images(
            image_unmasked,
            iimage,
            width=1000,
            height=int(1000*2/3),
        )
        print('max diff:', maxdiff)

    assert maxdiff < 0.5


def test_interpolate_image_and_noise_cs_gauss_circle_many(
        *,
        seed,
        sampling_rate=1.0,
        c=0.1,
        show=False):

    """
    test that our interpolation works decently for a linear
    piece missing from a gaussian image
    """

    rng = np.random.RandomState(seed=seed)

    flux_pdf = ngmix.priors.LogNormal(500, 500, rng=rng)
    hlr_pdf = ngmix.priors.LogNormal(0.5, 0.5, rng=rng)

    noise = 1

    dims = 500, 500
    imcen = (np.array(dims)-1)/2

    scale = 0.2

    buff = 50
    shift_max = (np.array(dims)-2*buff)/2.0 * scale

    nobj = 100
    weight = np.zeros(dims) + 1.0/noise**2
    noise_image = rng.normal(scale=noise, size=dims)

    badone = nobj//2

    subsize = 25

    psf = galsim.Moffat(fwhm=0.7, beta=2.5)

    objlist = []
    for i in range(nobj):

        shift_row, shift_col = rng.uniform(
            low=-shift_max, high=shift_max,
            size=2,
        )

        if i == badone:
            flux = 1000
            half_light_radius = 1.0e-4
        else:
            flux = flux_pdf.sample()
            half_light_radius = hlr_pdf.sample()

        obj0 = galsim.Exponential(
            half_light_radius=half_light_radius,
            flux=flux,
        )

        obj0 = obj0.shift(dx=shift_col, dy=shift_row)

        obj = galsim.Convolve(obj0, psf)
        objlist.append(obj)

        if i == badone:

            cen = imcen + (shift_row/scale, shift_col/scale)
            rows, cols = np.mgrid[
                0:dims[0],
                0:dims[1],
            ]
            rows = rows - cen[0]
            cols = cols - cen[1]

            radius2 = rows**2 + cols**2

            rowmin = int(cen[0]-subsize)
            rowmax = int(cen[0]+subsize) + 1
            colmin = int(cen[1]-subsize)
            colmax = int(cen[1]+subsize) + 1

            wbad = np.where(radius2 <= 3**2)

            weight[wbad] = 0.0

    objects = galsim.Sum(objlist)
    image_unmasked = objects.drawImage(
        nx=dims[1],
        ny=dims[0],
        scale=scale,
    ).array

    image_unmasked += rng.normal(scale=noise, size=image_unmasked.shape)

    # nbad = int(500 * 500 / 4)
    # xind = rng.choice(dims[1], size=nbad, replace=True)
    # yind = rng.choice(dims[0], size=nbad, replace=True)
    # weight[yind, xind] = 0.0

    wbad = np.where(weight <= 0.0)
    image_masked = image_unmasked.copy()
    image_masked[wbad] = 0.0

    bmask = np.zeros_like(image_unmasked, dtype=np.int32)
    bad_flags = 0

    _nlscale = np.sqrt(np.median(weight))

    def _scale(__im):
        return np.arcsinh(__im * _nlscale)

    def _descale(__im):
        return np.sinh(__im) / _nlscale

    _s_image_masked = _scale(image_masked)
    _s_noise = _scale(noise_image)

    tm0 = time.time()
    iimage, inoise = interpolate_image_and_noise_cs(
        image=_s_image_masked,
        weight=weight,
        bmask=bmask,
        bad_flags=bad_flags,
        noise=_s_noise,
        rng=np.random.RandomState(seed=45),
        sampling_rate=sampling_rate,
        c=c,
    )
    # iimage = _descale(iimage)
    # inoise = _descale(inoise)
    print('time:', time.time() - tm0)

    maxdiff = np.abs(image_unmasked-iimage).max()
    print('max diff:', maxdiff)

    if show:
        import images
        import matplotlib.pyplot as plt

        subim_um = _scale(image_unmasked)[
            rowmin:rowmax,
            colmin:colmax,
        ]

        """
        subim = image_masked[
            rowmin:rowmax,
            colmin:colmax,
        ]
        subwt = weight[
            rowmin:rowmax,
            colmin:colmax,
        ]
        images.view_mosaic([subim, subwt], labels=['im', 'weight'])
        """

        subinterp = iimage[
            rowmin:rowmax,
            colmin:colmax,
        ]
        images.compare_images(
            subim_um,
            subinterp,
            masked_image=image_masked[
                rowmin:rowmax,
                colmin:colmax,
            ],
            width=8,
        )
        plt.show()


if __name__ == '__main__':
    show = True
    for seed in range(413, 420):
        test_interpolate_image_and_noise_cs_gauss_circle_many(
            seed=seed,
            # sampling_rate=0.75,
            c=1,
            show=show,
        )

        if show:
            if input('hit a key: (q to quit): ') == 'q':
                raise KeyboardInterrupt('stopped')
    # test_interpolate_image_and_noise_cs_gauss_circle(show=True)
