import pytest
import esutil as eu


@pytest.fixture
def coadd_image_data():
    wcs = {
        'xtension': 'BINTABLE',
        'bitpix': 8,
        'naxis': 2,
        'naxis1': 24,
        'naxis2': 10000,
        'pcount': 83807884,
        'gcount': 1,
        'tfields': 3,
        'ttype1': 'COMPRESSED_DATA',
        'tform1': '1PB(8590)',
        'ttype2': 'ZSCALE  ',
        'tform2': '1D      ',
        'ttype3': 'ZZERO   ',
        'tform3': '1D      ',
        'zimage': True,
        'ztile1': 10000,
        'ztile2': 1,
        'zcmptype': 'RICE_ONE',
        'zname1': 'BLOCKSIZE',
        'zval1': 32,
        'zname2': 'BYTEPIX ',
        'zval2': 4,
        'zsimple': True,
        'zbitpix': -32,
        'znaxis': 2,
        'znaxis1': 10000,
        'znaxis2': 10000,
        'zextend': True,
        'extname': 'SCI     ',
        'equinox': 2000.0,
        'mjd-obs': 56545.15853046,
        'radesys': 'ICRS    ',
        'ctype1': 'RA---TAN',
        'cunit1': 'deg',
        'crval1': 320.688891,
        'crpix1': 5000.5,
        'cd1_1': -7.305555555556e-05,
        'cd1_2': 0.0,
        'ctype2': 'DEC--TAN',
        'cunit2': 'deg',
        'crval2': 0.016667,
        'crpix2': 5000.5,
        'cd2_1': 0.0,
        'cd2_2': 7.305555555556e-05}

    return {
        'wcs_header': wcs,
        'eu_wcs': eu.wcsutil.WCS(wcs),
        'position_offset': 1
    }
