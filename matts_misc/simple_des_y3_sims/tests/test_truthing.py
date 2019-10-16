import numpy as np
from ..truthing import make_coadd_grid_radec


def test_make_coadd_grid_radec(coadd_image_data):
    n_grid = 2
    rng = np.random.RandomState(seed=10)
    coadd_wcs = coadd_image_data['eu_wcs']

    ra, dec = make_coadd_grid_radec(
        n_grid=n_grid,
        coadd_wcs=coadd_wcs,
        rng=rng
    )

    # check them by hand
    rng = np.random.RandomState(seed=10)
    dL = 10000 / n_grid
    dL_2 = dL / 2

    _x = dL_2
    _y = dL_2
    _x += rng.uniform(-0.5, 0.5)
    _y += rng.uniform(-0.5, 0.5)
    _ra, _dec = coadd_wcs.image2sky(_x + 1, _y + 1)
    assert ra[0] == _ra
    assert dec[0] == _dec

    _x = dL_2 + dL
    _y = dL_2
    _x += rng.uniform(-0.5, 0.5)
    _y += rng.uniform(-0.5, 0.5)
    _ra, _dec = coadd_wcs.image2sky(_x + 1, _y + 1)
    assert ra[1] == _ra
    assert dec[1] == _dec

    _x = dL_2
    _y = dL_2 + dL
    _x += rng.uniform(-0.5, 0.5)
    _y += rng.uniform(-0.5, 0.5)
    _ra, _dec = coadd_wcs.image2sky(_x + 1, _y + 1)
    assert ra[2] == _ra
    assert dec[2] == _dec

    _x = dL_2 + dL
    _y = dL_2 + dL
    _x += rng.uniform(-0.5, 0.5)
    _y += rng.uniform(-0.5, 0.5)
    _ra, _dec = coadd_wcs.image2sky(_x + 1, _y + 1)
    assert ra[3] == _ra
    assert dec[3] == _dec
