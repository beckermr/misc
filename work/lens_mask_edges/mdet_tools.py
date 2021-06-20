import numpy as np
import tqdm
import yaml
from concurrent.futures import as_completed
from loky import get_reusable_executor
import ngmix
import galsim
import copy
from metadetect.detect import MEDSifier
from ngmix.gaussmom import GaussMom


CONFIG = yaml.safe_load("""\
  metacal:
    psf: fitgauss
    types: [noshear, 1p, 1m, 2p, 2m]
    use_noise_image: True

  psf:
    lm_pars:
      maxfev: 2000
      ftol: 1.0e-05
      xtol: 1.0e-05
    model: gauss

    # we try many times because if this fails we get no psf info
    # for the entire patch
    ntry: 10

  sx:

  weight:
    fwhm: 1.2  # arcsec

  meds:
    box_padding: 2
    box_type: iso_radius
    max_box_size: 53
    min_box_size: 33
    rad_fac: 2
    rad_min: 4

  # check for an edge hit
  bmask_flags: 1610612736  # 2**29 || 2**30

""")


def cut_nones(presults, mresults):
    """Cut entries that are None in a pair of lists. Any entry that is None
    in either list will exclude the item in the other.

    Parameters
    ----------
    presults : list
        One the list of things.
    mresults : list
        The other list of things.

    Returns
    -------
    pcut : list
        The cut list.
    mcut : list
        The cut list.
    """
    prr_keep = []
    mrr_keep = []
    for pr, mr in zip(presults, mresults):
        if pr is None or mr is None:
            continue
        prr_keep.append(pr)
        mrr_keep.append(mr)

    return prr_keep, mrr_keep


def _run_boostrap(x1, y1, x2, y2, wgts, verbose):
    rng = np.random.RandomState(seed=100)
    mvals = []
    cvals = []
    if verbose:
        itrl = tqdm.trange(500, leave=False, desc='running bootstrap', ncols=79)
    else:
        itrl = range(500)
    for _ in itrl:
        ind = rng.choice(len(y1), replace=True, size=len(y1))
        _wgts = wgts[ind].copy()
        _wgts /= np.sum(_wgts)
        mvals.append(np.mean(y1[ind] * _wgts) / np.mean(x1[ind] * _wgts) - 1)
        cvals.append(np.mean(y2[ind] * _wgts) / np.mean(x2[ind] * _wgts))

    return (
        np.mean(y1 * wgts) / np.mean(x1 * wgts) - 1, np.std(mvals),
        np.mean(y2 * wgts) / np.mean(x2 * wgts), np.std(cvals))


def _run_jackknife(x1, y1, x2, y2, wgts, jackknife, verbose):
    n_per = x1.shape[0] // jackknife
    n = n_per * jackknife
    x1j = np.zeros(jackknife)
    y1j = np.zeros(jackknife)
    x2j = np.zeros(jackknife)
    y2j = np.zeros(jackknife)
    wgtsj = np.zeros(jackknife)

    loc = 0
    if verbose:
        itrl = tqdm.trange(
            jackknife, desc='running jackknife sums', leave=False, ncols=79
        )
    else:
        itrl = range(jackknife)
    for i in itrl:
        wgtsj[i] = np.sum(wgts[loc:loc+n_per])
        x1j[i] = np.sum(x1[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]
        y1j[i] = np.sum(y1[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]
        x2j[i] = np.sum(x2[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]
        y2j[i] = np.sum(y2[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]

        loc += n_per

    mbar = np.mean(y1 * wgts) / np.mean(x1 * wgts) - 1
    cbar = np.mean(y2 * wgts) / np.mean(x2 * wgts)
    mvals = np.zeros(jackknife)
    cvals = np.zeros(jackknife)
    if verbose:
        itrl = tqdm.trange(
            jackknife, desc='running jackknife estimates', leave=False, ncols=79
        )
    else:
        itrl = range(jackknife)
    for i in itrl:
        _wgts = np.delete(wgtsj, i)
        mvals[i] = (
            np.sum(np.delete(y1j, i) * _wgts) / np.sum(np.delete(x1j, i) * _wgts)
            - 1
        )
        cvals[i] = (
            np.sum(np.delete(y2j, i) * _wgts) / np.sum(np.delete(x2j, i) * _wgts)
        )

    return (
        mbar,
        np.sqrt((n - n_per) / n * np.sum((mvals-mbar)**2)),
        cbar,
        np.sqrt((n - n_per) / n * np.sum((cvals-cbar)**2)),
    )


def estimate_m_and_c(
    presults,
    mresults,
    g_true,
    swap12=False,
    step=0.01,
    weights=None,
    jackknife=None,
    verbose=False,
):
    """Estimate m and c from paired lensing simulations.

    Parameters
    ----------
    presults : list of iterables or np.ndarray
        A list of iterables, each with g1p, g1m, g1, g2p, g2m, g2
        from running metadetect with a `g1` shear in the 1-component and
        0 true shear in the 2-component. If an array, it should have the named
        columns.
    mresults : list of iterables or np.ndarray
        A list of iterables, each with g1p, g1m, g1, g2p, g2m, g2
        from running metadetect with a -`g1` shear in the 1-component and
        0 true shear in the 2-component. If an array, it should have the named
        columns.
    g_true : float
        The true value of the shear on the 1-axis in the simulation. The other
        axis is assumd to havea true value of zero.
    swap12 : bool, optional
        If True, swap the roles of the 1- and 2-axes in the computation.
    step : float, optional
        The step used in metadetect for estimating the response. Default is
        0.01.
    weights : list of weights, optional
        Weights to apply to each sample. Will be normalized if not already.
    jackknife : int, optional
        The number of jackknife sections to use for error estimation. Default of
        None will do no jackknife and default to bootstrap error bars.
    verbose : bool, optional
        If True, print progress. Default is False.

    Returns
    -------
    m : float
        Estimate of the multiplicative bias.
    merr : float
        Estimat of the 1-sigma standard error in `m`.
    c : float
        Estimate of the additive bias.
    cerr : float
        Estimate of the 1-sigma standard error in `c`.
    """

    if isinstance(presults, list) or isinstance(mresults, list):
        prr_keep, mrr_keep = cut_nones(presults, mresults)

        def _get_stuff(rr):
            _a = np.vstack(rr)
            g1p = _a[:, 0]
            g1m = _a[:, 1]
            g1 = _a[:, 2]
            g2p = _a[:, 3]
            g2m = _a[:, 4]
            g2 = _a[:, 5]

            if swap12:
                g1p, g1m, g1, g2p, g2m, g2 = g2p, g2m, g2, g1p, g1m, g1

            return (
                g1, (g1p - g1m) / 2 / step * g_true,
                g2, (g2p - g2m) / 2 / step)

        g1p, R11p, g2p, R22p = _get_stuff(prr_keep)
        g1m, R11m, g2m, R22m = _get_stuff(mrr_keep)
    else:
        if swap12:
            g1p = presults["g2"]
            R11p = (presults["g2p"] - presults["g2m"]) / 2 / step * g_true
            g2p = presults["g1"]
            R22p = (presults["g1p"] - presults["g1m"]) / 2 / step

            g1m = mresults["g2"]
            R11m = (mresults["g2p"] - mresults["g2m"]) / 2 / step * g_true
            g2m = mresults["g1"]
            R22m = (mresults["g1p"] - mresults["g1m"]) / 2 / step
        else:
            g1p = presults["g1"]
            R11p = (presults["g1p"] - presults["g1m"]) / 2 / step * g_true
            g2p = presults["g2"]
            R22p = (presults["g2p"] - presults["g2m"]) / 2 / step

            g1m = mresults["g1"]
            R11m = (mresults["g1p"] - mresults["g1m"]) / 2 / step * g_true
            g2m = mresults["g2"]
            R22m = (mresults["g2p"] - mresults["g2m"]) / 2 / step

    if weights is not None:
        wgts = np.array(weights).astype(np.float64)
    else:
        wgts = np.ones(len(g1p)).astype(np.float64)
    wgts /= np.sum(wgts)

    msk = (
        np.isfinite(g1p) &
        np.isfinite(R11p) &
        np.isfinite(g1m) &
        np.isfinite(R11m) &
        np.isfinite(g2p) &
        np.isfinite(R22p) &
        np.isfinite(g2m) &
        np.isfinite(R22m))
    g1p = g1p[msk]
    R11p = R11p[msk]
    g1m = g1m[msk]
    R11m = R11m[msk]
    g2p = g2p[msk]
    R22p = R22p[msk]
    g2m = g2m[msk]
    R22m = R22m[msk]
    wgts = wgts[msk]

    x1 = (R11p + R11m)/2
    y1 = (g1p - g1m) / 2

    x2 = (R22p + R22m) / 2
    y2 = (g2p + g2m) / 2

    if jackknife:
        return _run_jackknife(x1, y1, x2, y2, wgts, jackknife, verbose)
    else:
        return _run_boostrap(x1, y1, x2, y2, wgts, verbose)


def make_obs(
    *,
    n_grid=6,
    dim=235,
    buff=20,
    scale=0.2,
    psf_fwhm=0.9,
    hlr=0.5,
    nse=1e-7,
    star_dxdy=117,
    star_rad=1,
    n_stars=5,
    seed=10,
    shear=(0.02, 0.0),
    mcal_shear=(0.0, 0.0)
):
    rng = np.random.RandomState(seed=seed)
    n_gals = n_grid**2
    tot_dim = dim + 2*buff
    tot_cen = (tot_dim-1)/2
    gloc = (np.arange(n_grid) + 0.5) * (dim / n_grid) - dim/2
    gloc *= scale
    dx, dy = np.meshgrid(gloc, gloc)
    dx = dx.ravel() + rng.uniform(low=-0.5, high=0.5, size=n_gals) * scale
    dy = dy.ravel() + rng.uniform(low=-0.5, high=0.5, size=n_gals) * scale
    ds = np.arange(n_gals) / (n_gals-1) * 0 + 1
    gals = galsim.Sum([
        galsim.Exponential(
            half_light_radius=hlr * _ds
        ).shift(
            _dx, _dy
        ).shear(
            g1=shear[0], g2=shear[1]
        ).shear(
            g1=mcal_shear[0], g2=mcal_shear[1]
        )
        for _ds, _dx, _dy in zip(ds, dx, dy)
    ])
    psf = galsim.Gaussian(fwhm=psf_fwhm)
    objs = galsim.Convolve([gals, psf])
    im = objs.drawImage(nx=tot_dim, ny=tot_dim, scale=scale).array

    im += rng.normal(size=im.shape, scale=nse)
    nim = rng.normal(size=im.shape, scale=nse)

    psf_dim = 53
    psf_cen = (psf_dim-1)/2
    psf_im = psf.drawImage(nx=psf_dim, ny=psf_dim, scale=scale).array

    # make bmask
    bmask = np.zeros_like(im, dtype=np.int32)
    x, y = np.meshgrid(np.arange(tot_dim), np.arange(tot_dim))
    sdata = []
    for _ in range(n_stars):
        sr2 = np.power(10.0, rng.uniform(low=star_rad, high=star_rad+0.2))**2
        sx = rng.uniform(low=-star_dxdy, high=star_dxdy) + tot_cen
        sy = rng.uniform(low=-star_dxdy, high=star_dxdy) + tot_cen
        dr2 = (x - sx)**2 + (y - sy)**2
        msk = dr2 < sr2
        bmask[msk] |= 2**0
        im[msk] = 0
        sdata.append((sx, sy, np.sqrt(sr2)))

    psf_obs = ngmix.Observation(
        image=psf_im,
        weight=np.ones_like(psf_im) / nse**2,
        jacobian=ngmix.DiagonalJacobian(scale=scale, row=psf_cen, col=psf_cen)
    )
    wgt = np.ones_like(im) / nse**2
    msk = bmask != 0
    wgt[msk] = 0.0
    mfrac = np.zeros_like(im)
    mfrac[msk] = 1.0
    obs = ngmix.Observation(
        image=im,
        noise=nim,
        weight=wgt,
        bmask=bmask,
        ormask=bmask,
        jacobian=ngmix.DiagonalJacobian(scale=scale, row=tot_cen, col=tot_cen),
        psf=psf_obs
    )
    obs.mfrac = mfrac
    mbobs = ngmix.MultiBandObsList()
    obsl = ngmix.ObsList()
    obsl.append(obs)
    mbobs.append(obsl)
    mbobs.meta["sdata"] = sdata
    return mbobs


def meas_mbmeds(mbobs, *, mask_width, maskflags=1, meds_config=None):
    # meas PSF
    mom = GaussMom(fwhm=1.2, obs=mbobs[0][0].psf)
    mom.go()
    res = mom.result
    psf_T = res['T']

    if meds_config is None:
        meds_config = copy.deepcopy(CONFIG["meds"])
    mfier = MEDSifier(
        mbobs,
        sx_config=None,
        meds_config=meds_config,
        maskflags=maskflags
    )
    mbmeds = mfier.get_multiband_meds()
    d = []
    dt = [
        ("flags", "i4"),
        ("g1", "f8"),
        ("g2", "f8"),
        ("s2n", "f8"),
        ("x", "f8"),
        ("y", "f8"),
        ('T_ratio', 'f8'),
    ]
    mw = mask_width
    for i, _mbobs in enumerate(mbmeds.get_mbobs_list()):
        if len(_mbobs) > 0 and len(_mbobs[0]) > 0:
            obs = _mbobs[0][0]
            cen = int((obs.bmask.shape[0]-1)/2)
            if np.any((obs.bmask[cen-mw:cen+mw+1, cen-mw:cen+mw+1] & maskflags) != 0):
                continue
            mom = GaussMom(fwhm=1.2, obs=obs)
            mom.go()
            res = mom.result
            if res["flags"] == 0:
                d.append((
                    res["flags"],
                    res["e"][0], res["e"][1],
                    res["s2n"],
                    mfier.cat["x"][i], mfier.cat["y"][i],
                    res['T'] / psf_T
                ))
            else:
                d.append((
                    res["flags"],
                    -9999, -9999,
                    -9999,
                    mfier.cat["x"][i], mfier.cat["y"][i],
                    -9999,
                ))
    return np.array(d, dtype=dt), mbobs


def _cut_cat(d):
    return d[
        (d["flags"] == 0)
        & (d["s2n"] > 1e4)
        & (d["T_ratio"] > 1.2)
    ]


def _meas_m(*, mask_width, **kwargs):
    step = 0.01
    _d, mbobs = meas_mbmeds(
        make_obs(shear=(0.02, 0.0), mcal_shear=(0, 0), **kwargs),
        mask_width=mask_width,
    )
    _d1p, mbobs1p = meas_mbmeds(
        make_obs(shear=(0.02, 0.0), mcal_shear=(step, 0), **kwargs),
        mask_width=mask_width,
    )
    _d1m, mbobs1m = meas_mbmeds(
        make_obs(shear=(0.02, 0.0), mcal_shear=(-step, 0), **kwargs),
        mask_width=mask_width,
    )
    _d = _cut_cat(_d)
    _d1p = _cut_cat(_d1p)
    _d1m = _cut_cat(_d1m)

    if len(_d) > 0 and len(_d1p) > 0 and len(_d1m) > 0:
        g1 = np.mean(_d["g1"])
        g1p = np.mean(_d1p["g1"])
        g1m = np.mean(_d1m["g1"])
        g2 = np.mean(_d["g2"])
        g2p = np.mean(_d1p["g2"])
        g2m = np.mean(_d1m["g2"])
        pres = (g1p, g1m, g1, g2p, g2m, g2)
    else:
        return None, None

    _d, mbobs = meas_mbmeds(
        make_obs(shear=(-0.02, 0.0), mcal_shear=(0, 0), **kwargs),
        mask_width=mask_width,
    )
    _d1p, mbobs1p = meas_mbmeds(
        make_obs(shear=(-0.02, 0.0), mcal_shear=(step, 0), **kwargs),
        mask_width=mask_width,
    )
    _d1m, mbobs1m = meas_mbmeds(
        make_obs(shear=(-0.02, 0.0), mcal_shear=(-step, 0), **kwargs),
        mask_width=mask_width,
    )
    _d = _cut_cat(_d)
    _d1p = _cut_cat(_d1p)
    _d1m = _cut_cat(_d1m)

    if len(_d) > 0 and len(_d1p) > 0 and len(_d1m) > 0:
        g1 = np.mean(_d["g1"])
        g1p = np.mean(_d1p["g1"])
        g1m = np.mean(_d1m["g1"])
        g2 = np.mean(_d["g2"])
        g2p = np.mean(_d1p["g2"])
        g2m = np.mean(_d1m["g2"])
        mres = (g1p, g1m, g1, g2p, g2m, g2)
    else:
        return None, None

    return pres, mres


def meas_m(*, mask_width, n_stars, n_jobs, seed, n_print=500):
    seeds = np.random.RandomState(seed=seed).randint(size=n_jobs, low=1, high=2**28)

    exe = get_reusable_executor()
    futs = [
        exe.submit(_meas_m, n_stars=n_stars, seed=s, mask_width=mask_width)
        for s in seeds
    ]
    pres = []
    mres = []
    n_done = 0
    for fut in tqdm.tqdm(as_completed(futs), total=len(futs), ncols=79):
        n_done += 1
        try:
            res = fut.result()
            pres.append(res[0])
            mres.append(res[1])
        except Exception as e:
            print(e)

        if n_done % n_print == 0:
            m, merr, c, cerr = estimate_m_and_c(
                pres,
                mres,
                0.02,
                jackknife=200 if n_done > 1000 else None,
            )
            mstr = "m +/- merr: %0.6f +/- %0.6f [10^(-3), 3sigma]" % (
                m/1e-3, 3*merr/1e-3)
            tqdm.tqdm.write(mstr)

            cstr = "c +/- cerr: %0.6f +/- %0.6f [10^(-5), 3sigma]" % (
                c/1e-3, 3*cerr/1e-3)
            tqdm.tqdm.write(cstr)

    m, merr, c, cerr = estimate_m_and_c(
        pres,
        mres,
        0.02,
        jackknife=200 if n_jobs > 1000 else None,
    )

    return m, merr, c, cerr, pres, mres
