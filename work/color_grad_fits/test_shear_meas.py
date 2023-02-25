import time
import copy
import numpy as np
import ngmix
import galsim
import metadetect
from esutil.pbar import PBar
import joblib


TEST_METADETECT_CONFIG = {
    "fitters": [
        {
            "model": "gauss",
            "weight": {"fwhm": 2.0},
            "symmetrize": False,
            "coadd": True,
            "wavg": False,
        },
    ],

    'metacal': {
        'psf': 'fitgauss',
        'types': ['noshear', '1p', '1m', '2p', '2m'],
    },

    'sx': None,

    'meds': {
        'min_box_size': 32,
        'max_box_size': 32,

        'box_type': 'iso_radius',

        'rad_min': 4,
        'rad_fac': 2,
        'box_padding': 2,

        'weight_type': "uberseg",
    },

    # check for an edge hit
    'bmask_flags': 2**30,

    'nodet_flags': 2**0,
}


def make_sim(
    *,
    seed,
    g1,
    g2,
    dim=251,
    buff=34,
    scale=0.25,
    dens=100,
    ngrid=7,
    snr=1e6,
    band_hlrs=(0.9, 0.75, 0.5),
    psf_fwhms=(1.1, 1.05, 0.9),
    nse_facs=(1.3, 1.0, 0.7),
):
    rng = np.random.RandomState(seed=seed)

    half_loc = (dim-buff*2)*scale/2

    if ngrid is None:
        area_arcmin2 = ((dim - buff*2)*scale/60)**2
        nobj = int(dens * area_arcmin2)
        x = rng.uniform(low=-half_loc, high=half_loc, size=nobj)
        y = rng.uniform(low=-half_loc, high=half_loc, size=nobj)
    else:
        half_ngrid = (ngrid-1)/2
        x, y = np.meshgrid(np.arange(ngrid), np.arange(ngrid))
        x = (x.ravel() - half_ngrid)/half_ngrid * half_loc
        y = (y.ravel() - half_ngrid)/half_ngrid * half_loc
        nobj = x.shape[0]

    uvs = []
    for ind in range(nobj):
        u, v = rng.uniform(low=-scale, high=scale, size=2)
        u += x[ind]
        v += y[ind]
        uvs.append((u, v))

    cen = (dim-1)/2
    psf_dim = 53
    psf_cen = (psf_dim-1)/2

    mbobs = ngmix.MultiBandObsList()
    for band_hlr, psf_fwhm, nse_fac in zip(band_hlrs, psf_fwhms, nse_facs):
        psf = galsim.Gaussian(fwhm=psf_fwhm)
        gals = []
        for ind in range(nobj):
            u, v = uvs[ind]
            gals.append(
                galsim.Exponential(half_light_radius=band_hlr).shift(u, v)
            )
        gals = galsim.Add(gals)
        gals = gals.shear(g1=g1, g2=g2)
        gals = galsim.Convolve([gals, psf])

        im = gals.drawImage(nx=dim, ny=dim, scale=scale).array
        psf_im = psf.drawImage(nx=psf_dim, ny=psf_dim, scale=scale).array

        nse = (
            np.sqrt(np.sum(
                galsim.Convolve([
                    psf,
                    galsim.Exponential(half_light_radius=band_hlr),
                ]).drawImage(scale=0.25).array**2)
            )
            / snr
        ) * nse_fac

        im += rng.normal(size=im.shape, scale=nse)
        wgt = np.ones_like(im) / nse**2
        jac = ngmix.DiagonalJacobian(scale=scale, row=cen, col=cen)
        psf_jac = ngmix.DiagonalJacobian(scale=scale, row=psf_cen, col=psf_cen)

        obs = ngmix.Observation(
            image=im,
            weight=wgt,
            jacobian=jac,
            ormask=np.zeros_like(im, dtype=np.int32),
            bmask=np.zeros_like(im, dtype=np.int32),
            psf=ngmix.Observation(
                image=psf_im,
                jacobian=psf_jac,
            ),
        )

        obslist = ngmix.ObsList()
        obslist.append(obs)
        mbobs.append(obslist)

    return mbobs


def _shear_cuts(arr, model):
    if model == "wmom":
        tmin = 1.2
    else:
        tmin = 0.5
    msk = (
        (arr[f'{model}_flags'] == 0)
        & (arr[f'{model}_s2n'] > 10)
        & (arr[f'{model}_T_ratio'] > tmin)
    )
    return msk


def _meas_shear_data(res, model):
    msk = _shear_cuts(res['noshear'], model)
    g1 = np.mean(res['noshear'][f'{model}_g'][msk, 0])
    g2 = np.mean(res['noshear'][f'{model}_g'][msk, 1])

    msk = _shear_cuts(res['1p'], model)
    g1_1p = np.mean(res['1p'][f'{model}_g'][msk, 0])
    msk = _shear_cuts(res['1m'], model)
    g1_1m = np.mean(res['1m'][f'{model}_g'][msk, 0])
    R11 = (g1_1p - g1_1m) / 0.02

    msk = _shear_cuts(res['2p'], model)
    g2_2p = np.mean(res['2p'][f'{model}_g'][msk, 1])
    msk = _shear_cuts(res['2m'], model)
    g2_2m = np.mean(res['2m'][f'{model}_g'][msk, 1])
    R22 = (g2_2p - g2_2m) / 0.02

    dt = [
        ('g1', 'f8'),
        ('g2', 'f8'),
        ('R11', 'f8'),
        ('R22', 'f8')]
    return np.array([(g1, g2, R11, R22)], dtype=dt)


def _bootstrap_stat(d1, d2, func, seed, nboot=500):
    dim = d1.shape[0]
    rng = np.random.RandomState(seed=seed)
    stats = []
    for _ in range(nboot):
        ind = rng.choice(dim, size=dim, replace=True)
        stats.append(func(d1[ind], d2[ind]))
    return stats


def meas_m_c_cancel(pres, mres):
    x = np.mean(pres['g1'] - mres['g1'])/2
    y = np.mean(pres['R11'] + mres['R11'])/2
    m = x/y/0.02 - 1

    x = np.mean(pres['g2'] + mres['g2'])/2
    y = np.mean(pres['R22'] + mres['R22'])/2
    c = x/y

    return m, c


def boostrap_m_c(pres, mres):
    m, c = meas_m_c_cancel(pres, mres)
    bdata = _bootstrap_stat(pres, mres, meas_m_c_cancel, 14324, nboot=500)
    merr, cerr = np.std(bdata, axis=0)
    return m, merr, c, cerr


def run_sim(seed, mdet_seed, coadd, wavg, **kwargs):
    mbobs_p = make_sim(seed=seed, g1=0.02, g2=0.0, **kwargs)
    cfg = copy.deepcopy(TEST_METADETECT_CONFIG)
    cfg["fitters"][0]["coadd"] = coadd
    cfg["fitters"][0]["wavg"] = wavg
    model = "gauss"
    _pres = metadetect.do_metadetect(
        copy.deepcopy(cfg),
        mbobs_p,
        np.random.RandomState(seed=mdet_seed)
    )
    if _pres is None:
        return None

    mbobs_m = make_sim(seed=seed, g1=-0.02, g2=0.0, **kwargs)
    _mres = metadetect.do_metadetect(
        copy.deepcopy(cfg),
        mbobs_m,
        np.random.RandomState(seed=mdet_seed)
    )
    if _mres is None:
        return None

    return _meas_shear_data(_pres, model), _meas_shear_data(_mres, model)


def test_shear_meas_simple():
    snr = 1e6
    ngrid = 7
    ntrial = ((100 // 48) + 1) * 48
    nsub = 48
    nitr = ntrial // nsub
    rng = np.random.RandomState(seed=116)
    seeds = rng.randint(low=1, high=2**29, size=ntrial)
    mdet_seeds = rng.randint(low=1, high=2**29, size=ntrial)

    tm0 = time.time()

    print("")

    pres_jnt = []
    mres_jnt = []
    pres_cd = []
    mres_cd = []
    loc = 0
    with joblib.Parallel(n_jobs=-1, verbose=100, backend='loky') as par:
        for itr in PBar(range(nitr)):
            for coadd in [True, False]:
                jobs = [
                    joblib.delayed(run_sim)(
                        seeds[loc+i],
                        mdet_seeds[loc+i],
                        coadd,
                        False,
                        snr=snr,
                        ngrid=ngrid,
                    )
                    for i in range(nsub)
                ]
                print("\n", end="", flush=True)
                outputs = par(jobs)

                for out in outputs:
                    if out is None:
                        continue
                    if coadd:
                        pres_cd.append(out[0])
                        mres_cd.append(out[1])
                    else:
                        pres_jnt.append(out[0])
                        mres_jnt.append(out[1])

                m, merr, c, cerr = boostrap_m_c(
                    np.concatenate(pres_cd if coadd else pres_jnt),
                    np.concatenate(mres_cd if coadd else mres_jnt),
                )
                print(
                    (
                        "\n"
                        "nsims: %d\n"
                        "coadd: %r\n"
                        "m [1e-3, 3sigma]: %s +/- %s\n"
                        "c [1e-5, 3sigma]: %s +/- %s\n"
                        "\n"
                    ) % (
                        len(pres_cd) if coadd else len(pres_jnt),
                        coadd,
                        m/1e-3,
                        3*merr/1e-3,
                        c/1e-5,
                        3*cerr/1e-5,
                    ),
                    flush=True,
                )

            loc += nsub

    total_time = time.time()-tm0
    print("time per:", total_time/ntrial, flush=True)

    for coadd in [True, False]:
        pres = np.concatenate(pres_cd if coadd else pres_jnt)
        mres = np.concatenate(mres_cd if coadd else mres_jnt)
        m, merr, c, cerr = boostrap_m_c(pres, mres)

        print(
            (
                "coadd: %r\n"
                "m [1e-3, 3sigma]: %s +/- %s"
                "\nc [1e-5, 3sigma]: %s +/- %s"
            ) % (
                coadd,
                m/1e-3,
                3*merr/1e-3,
                c/1e-5,
                3*cerr/1e-5,
            ),
            flush=True,
        )

        assert np.abs(m) < max(1e-3, 3*merr)
        assert np.abs(c) < 3*cerr
