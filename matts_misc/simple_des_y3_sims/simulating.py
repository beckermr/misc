import logging
import shutil
import os

import numpy as np
import yaml
import joblib
import galsim
import fitsio
from esutil.ostools import StagedOutFile

from .files import (
    get_band_info_file,
    make_dirs_for_file,
    get_truth_catalog_path,
    expand_path)
from .constants import MEDSCONF
from .truthing import make_coadd_grid_radec
from .sky_bounding import get_rough_sky_bounds, radec_to_uv
from .wcsing import get_esutil_wcs, get_galsim_wcs
from .galsiming import render_sources_for_image

logger = logging.getLogger(__name__)


class End2EndSimulation(object):
    """An end-to-end DES Y3 simulation.

    Parameters
    ----------
    seed : int
        The seed for the global RNG.
    output_meds_dir : str
        The output DEADATA/MEDS_DIR for the simulation data products.
    tilename : str
        The DES coadd tile to simulate.
    bands : str
        The bands to simulate.
    gal_kws : dict
        Keyword arguments to control the galaxy content of the simulation.
        Right now these should include:
            n_grid : int
                The galaxies will be put on a grid with `n_grid`
                on a side.
    psf_kws : dict
        Kyword arguments to control the PSF used for the simulation.
        Right now these should include:
            type : str
                One of 'gauss' and that's it.
    """
    def __init__(self, *,
                 seed, output_meds_dir, tilename, bands,
                 gal_kws, psf_kws):
        self.output_meds_dir = output_meds_dir
        self.tilename = tilename
        self.bands = bands
        self.gal_kws = gal_kws
        self.psf_kws = psf_kws
        self.seed = seed
        # any object within a 128 coadd pixel buffer of the edge of a CCD
        # will be rendered for that CCD
        self.bounds_buffer_uv = 128 * 0.263

        if self.psf_kws['type'] in ['piff', 'psfex']:
            self.draw_method = 'no_pixel'
        else:
            self.draw_method = 'auto'

        # make the RNGS
        # one for galaxies in the truth catalog
        # one for noise in the images
        seeds = np.random.RandomState(seed=seed).randint(
            low=1, high=2**30, size=2)
        self.gal_rng = np.random.RandomState(seed=seeds[0])
        self.noise_rng = np.random.RandomState(seed=seeds[1])

        # load the image info for each band
        self.info = {}
        for band in bands:
            fname = get_band_info_file(
                meds_dir=self.output_meds_dir,
                medsconf=MEDSCONF,
                tilename=self.tilename,
                band=band)
            with open(fname, 'r') as fp:
                self.info[band] = yaml.load(fp, Loader=yaml.Loader)

    def run(self):
        """Run the simulation w/ galsim, writing the data to disk."""

        logger.info(' simulating coadd tile %s', self.tilename)

        # step 1 - make the truth catalog
        truth_cat = self._make_truth_catalog()

        # step 2 - per band, write the images to a tile
        for band in self.bands:
            self._run_band(band=band, truth_cat=truth_cat)

    def _run_band(self, *, band, truth_cat):
        """Run a simulation of a truth cat for a given band."""

        logger.info(" rendering images in band %s", band)

        def obj_func(ind, truth_cat, pos):
            obj = galsim.Exponential(half_light_radius=0.5).withFlux(64000)
            psf = galsim.Gaussian(fwhm=0.9)
            obj = galsim.Convolve([obj, psf])
            return obj

        noise_seeds = self.noise_rng.randint(
            low=1, high=2**30, size=len(self.info[band]['src_info']))

        jobs = []
        for noise_seed, se_info in zip(
                noise_seeds, self.info[band]['src_info']):
            jobs.append(joblib.delayed(_render_se_image)(
                se_info=se_info,
                band=band,
                truth_cat=truth_cat,
                bounds_buffer_uv=self.bounds_buffer_uv,
                draw_method=self.draw_method,
                noise_seed=noise_seed,
                output_meds_dir=self.output_meds_dir,
                obj_func=obj_func))

        with joblib.Parallel(
                n_jobs=-1, backend='loky', verbose=50, max_nbytes=None) as p:
            p(jobs)

    def _make_truth_catalog(self):
        """Make the truth catalog."""
        # always done with first band
        band = self.bands[0]
        coadd_wcs = get_esutil_wcs(
            image_path=self.info[band]['image_path'],
            image_ext=self.info[band]['image_ext'])

        ra, dec, x, y = make_coadd_grid_radec(
            rng=self.gal_rng, coadd_wcs=coadd_wcs,
            return_xy=True, n_grid=self.gal_kws['n_grid'])

        truth_cat = np.zeros(
            len(ra), dtype=[
                ('number', 'i8'),
                ('ra', 'f8'),
                ('dec', 'f8'),
                ('x', 'f8'),
                ('y', 'f8')])
        truth_cat['number'] = np.arange(len(ra)).astype(np.int64) + 1
        truth_cat['ra'] = ra
        truth_cat['dec'] = dec
        truth_cat['x'] = x
        truth_cat['y'] = y

        truth_cat_path = get_truth_catalog_path(
            meds_dir=self.output_meds_dir,
            medsconf=MEDSCONF,
            tilename=self.tilename)

        make_dirs_for_file(truth_cat_path)
        fitsio.write(truth_cat_path, truth_cat, clobber=True)

        return truth_cat


def _render_se_image(
        *, se_info, band, truth_cat, bounds_buffer_uv,
        draw_method, noise_seed, output_meds_dir, obj_func):
    """Render an SE image.

    This function renders a full image and writes it to disk.

    Parameters
    ----------
    se_info : dict
        The entry from the `src_info` list for the coadd tile.
    band : str
        The band as a string.
    truth_cat : np.ndarray
        A structured array with the truth catalog. Must at least have the
        columns 'ra' and 'dec' in degrees.
    bounds_buffer_uv : float
        The buffer in arcseconds for finding sources in the image. Any source
        whose center lies outside of this buffer area around the CCD will not
        be rendered for that CCD.
    draw_method : str
        The method used to draw the image. See the docs of `GSObject.drawImage`
        for details and options. Usually 'auto' is correct unless using a
        PSF with the pixel in which case 'no_pixel' is the right choice.
    noise_seed : int
        The RNG seed to use to generate the noise field for the image.
    output_meds_dir : str
        The output DEADATA/MEDS_DIR for the simulation data products.
    obj_func : callable
        A function with signature `obj_func(src_ind, truth_cat, pos)` that
        returns the galsim object to be rendered for a given index of the
        truth catalog and its image position `pos`.
    """

    # step 1 - get the set of good objects for the CCD
    msk_inds = _cut_tuth_cat_to_se_image(
        truth_cat=truth_cat,
        se_info=se_info,
        bounds_buffer_uv=bounds_buffer_uv)

    # step 2 - render the objects
    im = _render_all_objects(
        msk_inds=msk_inds,
        truth_cat=truth_cat,
        se_info=se_info,
        band=band,
        obj_func=obj_func,
        draw_method=draw_method)

    # step 3 - add bkg and noise
    # also removes the zero point
    im, wgt, bkg = _add_noise_and_background(
        image=im,
        se_info=se_info,
        noise_seed=noise_seed)

    # step 4 - write to disk
    _write_se_img_wgt_bkg(
        image=im,
        weight=wgt,
        background=bkg,
        se_info=se_info,
        output_meds_dir=output_meds_dir)


def _cut_tuth_cat_to_se_image(*, truth_cat, se_info, bounds_buffer_uv):
    """get the inds of the objects to render from the truth catalog"""
    wcs = get_esutil_wcs(
        image_path=se_info['image_path'],
        image_ext=se_info['image_ext'])
    sky_bnds, ra_ccd, dec_ccd = get_rough_sky_bounds(
        im_shape=se_info['image_shape'],
        wcs=wcs,
        position_offset=se_info['position_offset'],
        bounds_buffer_uv=bounds_buffer_uv,
        n_grid=4)
    u, v = radec_to_uv(truth_cat['ra'], truth_cat['dec'], ra_ccd, dec_ccd)
    sim_msk = sky_bnds.contains_points(u, v)
    msk_inds, = np.where(sim_msk)
    return msk_inds


def _render_all_objects(
        *, msk_inds, truth_cat, se_info, band, obj_func, draw_method):
    gs_wcs = get_galsim_wcs(
        image_path=se_info['image_path'],
        image_ext=se_info['image_ext'])

    def _src_func(ind):
        pos = gs_wcs.toImage(galsim.CelestialCoord(
            ra=truth_cat['ra'][ind] * galsim.degrees,
            dec=truth_cat['dec'][ind] * galsim.degrees))

        obj = obj_func(ind, truth_cat, pos)

        return obj, pos

    im = render_sources_for_image(
        image_shape=se_info['image_shape'],
        wcs=gs_wcs,
        draw_method=draw_method,
        src_inds=msk_inds,
        src_func=_src_func,
        n_jobs=1)

    return im.array


def _add_noise_and_background(*, image, se_info, noise_seed):
    """add noise and background to an image, remove the zero point"""

    noise_rng = np.random.RandomState(seed=noise_seed)

    # first back to ADU units
    image /= se_info['scale']

    # add the background
    bkg = fitsio.read(se_info['bkg_path'], ext=se_info['bkg_ext'])
    image += bkg

    # now add noise
    wgt = fitsio.read(se_info['weight_path'], ext=se_info['weight_ext'])
    bmask = fitsio.read(se_info['bmask_path'], ext=se_info['bmask_ext'])
    img_std = 1.0 / np.sqrt(np.median(wgt[bmask == 0]))
    image += (noise_rng.normal(size=image.shape) * img_std)
    wgt[:, :] = 1.0 / img_std**2

    return image, wgt, bkg


def _write_se_img_wgt_bkg(
        *, image, weight, background, se_info, output_meds_dir):
    # these should be the same
    assert se_info['image_path'] == se_info['weight_path'], se_info
    assert se_info['image_path'] == se_info['bmask_path'], se_info

    # and not this
    assert se_info['image_path'] != se_info['bkg_path']

    # get the final image file path and write
    image_file = se_info['image_path'].replace(
        '$MEDS_DIR', output_meds_dir)
    make_dirs_for_file(image_file)
    with StagedOutFile(
            image_file, tmpdir=os.environ.get('TMPDIR', None)) as sf:
        # copy to the place we stage from
        shutil.copy(expand_path(se_info['image_path']), sf.path)

        # open in read-write mode and replace the data
        with fitsio.FITS(sf.path, mode='rw') as fits:
            fits[se_info['image_ext']].write(image)
            fits[se_info['weight_ext']].write(weight)
            fits[se_info['bmask_ext']].write(
                np.zeros_like(image, dtype=np.int16))

    # get the background file path and write
    bkg_file = se_info['bkg_path'].replace(
        '$MEDS_DIR', output_meds_dir)
    make_dirs_for_file(bkg_file)
    with StagedOutFile(bkg_file, tmpdir=os.environ.get('TMPDIR', None)) as sf:
        # copy to the place we stage from
        shutil.copy(expand_path(se_info['bkg_path']), sf.path)

        # open in read-write mode and replace the data
        with fitsio.FITS(sf.path, mode='rw') as fits:
            fits[se_info['bkg_ext']].write(background)
