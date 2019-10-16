import numpy as np
import galsim
import pytest

from ..galsiming import render_source_in_image, render_sources_for_image


@pytest.mark.parametrize('x', [55, 55.5, 55.3])
@pytest.mark.parametrize('y', [50, 50.5, 50.1])
def test_render_source_in_image(x, y):
    source = galsim.Gaussian(fwhm=0.9)
    local_wcs = galsim.PixelScale(0.25)
    draw_method = 'auto'
    image_pos = galsim.PositionD(x=x, y=y)

    expected_shape = source.drawImage(
        wcs=local_wcs, method=draw_method).array.shape

    stamp = render_source_in_image(
        source=source,
        image_pos=image_pos,
        local_wcs=local_wcs,
        draw_method=draw_method
    )

    assert stamp.array.shape == expected_shape

    # any odd number works here for the image size - need to have image center
    # be a pixel center
    full_img = source.shift(
        dx=image_pos.x*0.25, dy=image_pos.y*0.25
    ).drawImage(
        wcs=local_wcs, nx=187, ny=187)
    # since the shift method offsets from image center, we set
    # image center to (0, 0) so the object shifts match the expected pixel
    # location
    full_img.setCenter(0, 0)

    # now the bounds of the stamp are in the same coords as the full image
    # so just grab that subset and compare
    assert np.allclose(
        full_img[stamp.bounds].array, stamp.array, rtol=0, atol=5e-7)


def test_render_sources_for_image():
    image_shape = (64, 64)
    wcs = galsim.PixelScale(0.25)
    draw_method = 'auto'
    src_inds = np.arange(5)

    def src_func(ind):
        obj = galsim.Gaussian(fwhm=ind*0.1 + 0.9)
        rng = np.random.RandomState(seed=ind+1)
        image_pos = galsim.PositionD(
            x=rng.uniform(low=20, high=40),
            y=rng.uniform(low=20, high=40),
        )
        return obj, image_pos

    im = render_sources_for_image(
        image_shape=image_shape,
        wcs=wcs,
        draw_method=draw_method,
        src_inds=src_inds,
        src_func=src_func,
        n_jobs=1)

    # now render them via direct sum of objects
    objs = []
    for ind in src_inds:
        obj, pos = src_func(ind)
        objs.append(obj.shift(dx=pos.x*0.25, dy=pos.y*0.25))
    im_sum = galsim.Sum(objs).drawImage(wcs=wcs, nx=155, ny=155)
    # we set the center to (0, 0) so that the offsets from the center
    # match the coords of the objects
    im_sum.setCenter(0, 0)

    assert np.allclose(im_sum[im.bounds].array, im.array, rtol=0, atol=5e-7)
