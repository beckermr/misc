import os
import tempfile

import fitsio
import numpy as np

from ..simulating import (
    _add_noise_and_background,
    _cut_tuth_cat_to_se_image)


def test_add_noise_and_background():
    with tempfile.TemporaryDirectory() as tmpdir:
        img_pth = os.path.join(tmpdir, 'img.fits')

        se_info = {}
        se_info['image_shape'] = (11, 13)
        se_info['scale'] = 34.5
        se_info['bkg_path'] = img_pth
        se_info['bkg_ext'] = 'a'
        se_info['weight_path'] = img_pth
        se_info['weight_ext'] = 'b'
        se_info['bmask_path'] = img_pth
        se_info['bmask_ext'] = 'c'

        rng = np.random.RandomState(seed=10)
        img = rng.normal(size=se_info['image_shape'])
        bkg = rng.normal(size=se_info['image_shape'])
        wgt = np.exp(rng.normal(size=se_info['image_shape']))
        bmask = (rng.uniform(size=se_info['image_shape']) < 0.1).astype(
            np.int16)

        with fitsio.FITS(img_pth, 'rw', clobber=True) as fits:
            fits.write(bkg, extname='a')
            fits.write(wgt, extname='b')
            fits.write(bmask, extname='c')

        # manipulations done in place so make the truth here
        noise_rng = np.random.RandomState(seed=145)
        nse = noise_rng.normal(size=img.shape)
        img_std = 1.0 / np.sqrt(np.median(wgt[bmask == 0]))
        timg = img / se_info['scale'] + bkg + (nse * img_std)

        rimg, rwgt, rbkg = _add_noise_and_background(
            image=img,
            se_info=se_info,
            noise_seed=145)

        assert np.array_equal(rbkg, bkg)
        assert np.allclose(rwgt, 1.0 / img_std**2)
        assert np.allclose(rimg, timg)


def test_cut_tuth_cat_to_se_image(coadd_image_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        img = os.path.join(tmpdir, 'img.fits')
        with fitsio.FITS(img, 'rw', clobber=True) as fits:
            fits.write(
                np.zeros((10000, 10000), dtype=np.float32),
                extname='a',
                header=coadd_image_data['wcs_header'])

        se_info = {}
        se_info['image_shape'] = (10000, 10000)
        se_info['image_path'] = img
        se_info['image_ext'] = 'a'
        se_info['position_offset'] = 1

        wcs = coadd_image_data['eu_wcs']
        truth_cat = np.zeros(2, dtype=[('ra', 'f8'), ('dec', 'f8')])

        ra, dec = wcs.image2sky(5000, 5000)
        truth_cat['ra'][0] = ra
        truth_cat['dec'][0] = dec

        ra, dec = wcs.image2sky(-500, -500)
        truth_cat['ra'][1] = ra
        truth_cat['dec'][1] = dec

        msk_inds = _cut_tuth_cat_to_se_image(
            truth_cat=truth_cat,
            se_info=se_info,
            bounds_buffer_uv=128 * 0.263)
        assert np.array_equal(msk_inds, np.array([0]))
        assert len(msk_inds) == 1
