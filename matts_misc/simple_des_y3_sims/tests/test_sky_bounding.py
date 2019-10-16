import pytest

import numpy as np
import esutil as eu

from ..sky_bounding import get_rough_sky_bounds, radec_to_uv


@pytest.fixture
def bnds_data():
    wcs = eu.wcsutil.WCS(dict(
        naxis1=2048,
        naxis2=4096,
        equinox=2000.00000000,
        radesys='ICRS    ',
        ctype1='RA---Tpv',
        ctype2='DEC--Tpv',
        cunit1='deg     ',
        cunit2='deg     ',
        crval1=3.204939506590E+02,
        crval2=6.143643163701E-01,
        crpix1=1.116880000000E+04,
        crpix2=-4.341000000000E+03,
        cd1_1=-1.508740062706E-07,
        cd1_2=7.285654526391E-05,
        cd2_1=-7.285164861781E-05,
        cd2_2=-1.494473487673E-07,
        pv1_0=-3.490695800309E-03,
        pv1_1=1.014642131611E+00,
        pv1_2=6.838988292402E-03,
        pv1_4=-1.064604424650E-02,
        pv1_5=-1.671893935475E-02,
        pv1_6=-7.265386507767E-03,
        pv1_7=1.377291873667E-03,
        pv1_8=8.590284815313E-03,
        pv1_9=5.428141605600E-03,
        pv1_10=2.582395791405E-03,
        pv2_0=-4.565360201645E-03,
        pv2_1=1.015176020471E+00,
        pv2_2=1.033041271404E-02,
        pv2_4=-1.145876554837E-02,
        pv2_5=-2.100570193619E-02,
        pv2_6=-9.155231884000E-03,
        pv2_7=1.907737076892E-03,
        pv2_8=1.141155239105E-02,
        pv2_9=6.262968730573E-03,
        pv2_10=2.846317319947E-03))
    position_offset = 1
    sky_bnds, ra_ccd, dec_ccd = get_rough_sky_bounds(
        im_shape=wcs.get_naxis()[::-1],
        wcs=wcs,
        position_offset=position_offset,
        bounds_buffer_uv=16.0,
        n_grid=4)

    return wcs, position_offset, sky_bnds, ra_ccd, dec_ccd


def test_get_rough_sky_bounds_smoke(bnds_data):
    wcs, position_offset, sky_bnds, ra_ccd, dec_ccd = bnds_data
    ncol, nrow = wcs.get_naxis()
    row, col = np.mgrid[0:nrow+64:64, 0:ncol+64:64]
    row = row.ravel()
    col = col.ravel()

    ra, dec = wcs.image2sky(
        x=col + position_offset,
        y=row + position_offset)
    u, v = radec_to_uv(ra, dec, ra_ccd, dec_ccd)
    assert np.all(sky_bnds.contains_points(u, v))

    # dither things a bit too
    for col_dither in [-0.51, 0.51]:
        for row_dither in [-0.51, 0.52]:
            ra, dec = wcs.image2sky(
                x=col + position_offset + col_dither,
                y=row + position_offset + row_dither)
            u, v = radec_to_uv(ra, dec, ra_ccd, dec_ccd)
            assert np.all(sky_bnds.contains_points(u, v))


@pytest.mark.parametrize(
    'offset,outside',
    [(45, False),
     (75, True)])
def test_get_rough_sky_bounds_edge_buffer(bnds_data, offset, outside):
    # the buffer above is 16 arcsec
    # the wcs above is from DES
    # thus we want 16 / 0.263 ~ 61 pixels
    # so we test inside at 45 and outside at 75

    wcs, position_offset, sky_bnds, ra_ccd, dec_ccd = bnds_data

    ncol, nrow = wcs.get_naxis()
    row_t = np.linspace(0, nrow, (nrow + 64) // 64)
    col_t = np.ones(nrow // 64 + 1) * 0
    row_b = np.linspace(0, nrow, (nrow + 64) // 64)
    col_b = np.ones(nrow // 64 + 1) * ncol

    row_l = np.ones(ncol // 64 + 1) * 0
    col_l = np.linspace(0, ncol, (ncol + 64) // 64)
    row_r = np.ones(ncol // 64 + 1) * nrow
    col_r = np.linspace(0, ncol, (ncol + 64) // 64)

    def _test_it(row, col, col_dither, row_dither, outside=False):
        ra, dec = wcs.image2sky(
            x=col + position_offset + col_dither,
            y=row + position_offset + row_dither)
        u, v = radec_to_uv(ra, dec, ra_ccd, dec_ccd)
        if outside:
            assert np.all(~sky_bnds.contains_points(u, v))
        else:
            assert np.all(sky_bnds.contains_points(u, v))

    _test_it(row_t, col_t, -offset, 0, outside=outside)
    _test_it(row_b, col_b, offset, 0, outside=outside)
    _test_it(row_l, col_l, 0, -offset, outside=outside)
    _test_it(row_r, col_r, 0, offset, outside=outside)
