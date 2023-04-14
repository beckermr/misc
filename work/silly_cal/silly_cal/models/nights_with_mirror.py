"""
This module models a survey zero point as a sum of three terms

    zp_exposure =
        sqrt(rho) * zp_night
        + sqrt(1-rho) * zp_noise
        + tel_eff * (time since last cleaning)

The model is that there is a correlated zp for the night
and some random noise for each exposure. The correlation
coefficient is rho. These terms get added onto a background
of a slowly changing overall efficiency since the last time
the system was cleaned.

To use this code, you do the following

    fake_data = gen_fake_data(edata=edata, nside=128, seed=10)
    guess = gen_guess(fake_data["opt_kwargs"])
    grad = np.zeros_like(guess)
    mygrads = np.zeros((numba.get_num_threads(), pars.shape[0]))
    chi2_mean, grad = value_and_grad(pars, grad mygrads, **fake_data["opt_kwargs"])

"""  # noqa
import numpy as np
import smatch
import hpgeom
import numba


def gen_fake_data(
    *, edata, nside, seed,
    focal_plane_radius=1.0,
    rho_night=0.891,
    zp_std=0.0327,
    tel_eff_mean=-1.51e-4,
    tel_eff_std=5e-6,
    star_nse_std=0.1,
    sstar_nse_std=2e-3,
    target_nstar=None,
    period=1095,
):
    """Generate fake data for a model with nightly zp correlations and
    a slow background change due to system cleanings.

    The model is

        zp_exposure =
            sqrt(rho) * zp_night
            + sqrt(1-rho) * zp_noise
            + tel_eff * (time since last cleaning)

    The model is that there is a correlated zp for the night
    and some random noise for each exposure. The correlation
    coefficient is rho. These terms get added onto a background
    of a slowly changing overall efficiency since the last time
    the system was cleaned.

    This function generates observations of the stars on a healpix
    grid of nside. It randomly picks on of those stars to be the
    calspec standard star. It then generates all of the observations
    of each star and outputs the data for the model.

    The default parameters here are set to those that rouhgly
    match zero-point data from DES Y6 FGCM results.

    Parameters
    ----------
    edata : structred array-like
        A structured array or similar with columns "mjd_obs", "ra", "dec"
        giving the MJD, ra and dec of each exposure. Units should be days
        and degrees.
    nside : int
        The nside for the star healpix grid.
    seed : int
        The seed for the RNG.
    focal_plane_radius : float, optional
        The focal plane is assumed to be a circle of this angular size in
        degrees. Default is 1.0 which is roughly correct for the Blanco+DECam.
    rho_night : float, optional
        The correlation between exposure zero-points on the same night. The
        default is 0.891 which matches a rough estimate from DES Y6 FGCM.
    zp_std : float, optional
        The total noise in the zero-points. The default is 0.0327 magnitudes
        which matches a rough estimate from DES Y6 FGCM.
    tel_eff_mean : float, optional
        The change in telescope efficiency in mags per dat. Default of -1.51e-4
        is taken from DES Y6 FGCM.
    tel_eff_std : float, optional
        The scatter in the telescope efficiency in mags per day. I invented the
        default of 5e-6.
    star_nse_std : float, optional
        The observational noise on each star. The default of 0.1 is roughly S/N of
        10.
    sstar_nse_std : float, optional
        The observational noise on the calspec standard. The default of 2e-3 is taken
        from DES Y6 FGCM.
    target_nstar : float, option
        If given, the magnitude errors on star observations are reduced to
        simulate observations of `target_nstar` stars. For DES Y6 FGCM, this is
        roughly 20e6.
    period : int, optional
        The frequency in days with which the system is cleaned. The default
        is three years or 1095 days.

    Returns
    -------
    fake_data : dict
        A dictionary with two keys

            params : dict of the input parameters, see docs above
            data : dict of the output data to use to constrain the model

                "night" : the night of each exposure
                "unight" : the unique nights in the data
                "inv_unight" : an array such that night = unight[inv_unight]
                "uyear" : the unique "years" in the exposures - these are really the
                    unique periods given by `period` starting from the first night
                "inv_uyear" :  an array such that year = uyear[inv_uyear]
                "night_in_year" : the night in the year for each exposure
                "true_zp_night" : the true nightly contribution to the zp
                "true_zp_nse" : the true per exposure noise contribution to the zp
                "true_tel_eff" the true efficiency of the telescope for each period
                "true_zp" : the true zero point of each exposure
                "true_star" : the true mag of each star
                "true_star_nest_ind" : the healpix nest index of each star
                "ied" : the index into the list of exposures of each star observation
                "istar" : the index into the list of stars of each star observation
                "isstar" : the index into the list of stars of the calspec star
                "star_ra" : the ra of each star
                "star_dec" : the dec of each star
                "star_obs" : the observed magnitude of each star observation
                "star_obs_err" : the magnitude error of each star observaton
                "rho_night" : the nightly zp correlation parameter

            opt_kwargs = dict of data for the optimizer
                This is a subset of the above plus some dimensions of certain arrays.
    """  # noqa
    rng = np.random.RandomState(seed=seed)

    period = 365*3
    night = (edata["mjd_obs"] + 0.5).astype(int)
    night = night - np.min(night) + 100
    year = night // period
    night_in_year = (night % period)/period
    uyear, inv_uyear = np.unique(year, return_inverse=True)
    unight, inv_unight = np.unique(night, return_inverse=True)

    true_zp_night = rng.normal(
        scale=zp_std,
        size=unight.shape,
    )

    true_zp_nse = rng.normal(
        scale=zp_std,
        size=edata.shape,
    )

    width = np.sqrt(12) * tel_eff_std
    true_tel_eff = rng.uniform(
        low=tel_eff_mean - width/2,
        high=tel_eff_mean + width/2,
        size=uyear.shape,
    ) * 365.0

    rho_night_fac = np.sqrt(1.0 - rho_night)
    sqrt_rho_night = np.sqrt(rho_night)

    true_zp = (
        sqrt_rho_night * true_zp_night[inv_unight]
        + rho_night_fac * true_zp_nse
        + true_tel_eff[inv_uyear] * night_in_year
    )

    edmatch = smatch.Matcher(edata["ra"], edata["dec"])
    star_ra, star_dec = hpgeom.pixel_to_angle(
        nside, np.arange(hpgeom.nside_to_npixel(nside))
    )

    _, _, nest_ind, _ = edmatch.query_radius(
        star_ra, star_dec, focal_plane_radius, return_indices=True
    )
    nest_ind = np.unique(nest_ind)

    star_ra = star_ra[nest_ind]
    star_dec = star_dec[nest_ind]

    if target_nstar is not None:
        err_fac = np.sqrt(star_ra.shape[0]/target_nstar)
        star_nse_std *= err_fac
        sstar_nse_std *= err_fac

    _, ied, istar, _ = edmatch.query_radius(
        star_ra, star_dec, focal_plane_radius, return_indices=True
    )

    true_star = np.zeros_like(star_ra)

    star_obs = (
        true_star[istar]
        + true_zp[ied]
        + rng.normal(
            scale=star_nse_std,
            size=istar.shape,
        )
    )
    star_obs_err = np.ones_like(star_obs) * star_nse_std

    isstar = rng.choice(len(star_ra))
    star_obs = np.concatenate(
        [
            star_obs,
            true_star[isstar]
            + rng.normal(
                scale=sstar_nse_std,
                size=1,
            )
        ],
        axis=0,
    )
    star_obs_err = np.concatenate(
        [
            star_obs_err,
            [sstar_nse_std],
        ],
        axis=0,
    )

    data = {
        "params": {
            "nside": nside,
            "seed": seed,
            "zp_std": zp_std,
            "star_nse_std": star_nse_std,
            "sstar_nse_std": sstar_nse_std,
            "rho_night": rho_night,
            "tel_eff_mean": tel_eff_mean,
            "tel_eff_std": tel_eff_std,
            "period": period,
            "focal_plane_radius": focal_plane_radius,
        },
        "data": {
            "night": night,
            "unight": unight,
            "inv_unight": inv_unight,
            "uyear": uyear,
            "inv_uyear": inv_uyear,
            "night_in_year": night_in_year,
            "true_zp_night": true_zp_night,
            "true_zp_nse": true_zp_nse,
            "true_tel_eff": true_tel_eff,
            "true_zp": true_zp,
            "true_star": true_star,
            "true_star_nest_ind": nest_ind,
            "ied": ied,
            "istar": istar,
            "isstar": isstar,
            "star_ra": star_ra,
            "star_dec": star_dec,
            "star_obs": star_obs,
            "star_obs_err": star_obs_err,
            "rho_night": rho_night,
        }
    }
    data["opt_kwargs"] = dict(
        nnight=data["data"]["true_zp_night"].shape[0],
        nexp=data["data"]["true_zp_nse"].shape[0],
        nstar=data["data"]["true_star"].shape[0],
        nyear=data["data"]["true_tel_eff"].shape[0],
        inv_uyear=data["data"]["inv_uyear"],
        night_in_year=data["data"]["night_in_year"],
        inv_unight=data["data"]["inv_unight"],
        istar=data["data"]["istar"],
        ied=data["data"]["ied"],
        isstar=data["data"]["isstar"],
        star_obs=data["data"]["star_obs"],
        star_obs_err=data["data"]["star_obs_err"],
        rho_night=data["params"]["rho_night"],
    )
    return data


def gen_guess(opt_kwargs, eps=1e-4, rng=None):
    """Generate a guess of the parameters of the model.

    Parameters
    ----------
    opt_kwargs : dict
        The keyword arguments to the optimizer function.
    eps : float, optional
        The scale of the guess.
    rng : np.random.RandomState, int or None
        If None, the default rng is used. Otherwise, one is made
        with the int seed or the input one is used.

    Returns
    -------
    guess : array-like
        The guess at the parameters.
    """
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(seed=rng)

    nump = (
        opt_kwargs["nyear"]
        + opt_kwargs["nnight"]
        + opt_kwargs["nexp"]
        + opt_kwargs["nstar"]
    )
    g = rng.normal(scale=eps, size=nump)
    g[0:opt_kwargs["nyear"]] = -1.5e-4 * (1.0 + g[0:opt_kwargs["nyear"]]) * 365
    return g


@numba.njit(parallel=True, fastmath=True, nogil=True)
def value_and_grad(
    pars, grad, mygrads,
    *, nyear, nnight, nexp, nstar,
    inv_uyear,
    inv_unight,
    istar, ied, isstar,
    night_in_year,
    star_obs, star_obs_err,
    rho_night,
):
    """Compute the chi2/nobs for this model.

    Parameters
    ----------
    pars : array-like
        The paramneters.
    grad : array-like
        A work array of shape (npars,)
    mygrads : array-like
        A work array of shape (numba.get_num_threads(), npars).
    **opt_kwargs : unloaded dict
        The "opt_kwargs" key of the fake data as keywords.

    Returns
    -------
    mean_chi2 : float
        chi2/nobs for the parameters.
    grad : array-like
        The array of derivatives of the chi2/nobs wrt the parameters.
    """
    nt = numba.get_num_threads()
    tid = numba.get_thread_id()

    grad[:] = 0.0
    chi2 = 0.0
    nobs = star_obs.shape[0]

    tel_eff = pars[0:nyear]
    zp_night = pars[nyear:nyear+nnight]
    zp_exp = pars[nyear+nnight:nyear+nnight+nexp]
    true_star = pars[nyear+nnight+nexp:]

    rho_night_fac = np.sqrt(1.0 - rho_night)
    sqrt_rho_night = np.sqrt(rho_night)

    mygrads[:, :] = 0.0

    for i in numba.prange(nobs-1):
        _istar = istar[i]
        _ied = ied[i]
        _inight = inv_unight[_ied]
        _iyear = inv_uyear[_ied]

        pred_zp = (
            sqrt_rho_night * zp_night[_inight]
            + rho_night_fac * zp_exp[_ied]
            + night_in_year[_ied] * tel_eff[_iyear]
        )
        pred_star = true_star[_istar] + pred_zp

        chi = (star_obs[i] - pred_star) / star_obs_err[i]
        chi2 += (chi*chi)

        efac = -2.0 * chi / star_obs_err[i] / nobs

        mygrads[tid, _iyear] += efac * night_in_year[_ied]
        mygrads[tid, nyear + _inight] += efac * sqrt_rho_night
        mygrads[tid, nyear + nnight + _ied] += efac * rho_night_fac
        mygrads[tid, nyear + nnight + nexp + _istar] += efac

    for i in range(nt):
        grad += mygrads[i]

    i = nobs-1
    pred_star = true_star[isstar]
    chi = (star_obs[i] - pred_star) / star_obs_err[i]
    chi2 += (chi*chi)
    efac = -2.0 * chi / star_obs_err[i] / nobs
    grad[nyear + nnight + nexp + isstar] += efac

    return chi2/nobs, grad
