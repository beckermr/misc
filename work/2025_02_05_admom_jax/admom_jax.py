from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom

LOW_DET_VAL = 1.0e-200


def _fwhm_to_T(fwhm):
    sigma = fwhm / 2.3548200450309493
    return 2 * sigma ** 2


def _get_jac_scale(dudx, dudy, dvdx, dvdy):
    return jnp.sqrt(jnp.abs(dvdy*dudx - dvdx*dudy))


def _e2mom(e1, e2, T):
    T_2 = 0.5 * T
    irc = e2 * T_2
    icc = (1 + e1) * T_2
    irr = (1 - e1) * T_2

    return irr, irc, icc


def _mom2e(irr, irc, icc):
    T = irr + icc
    e1 = (icc - irr) / T
    e2 = 2.0 * irc / T

    return e1, e2, T


def _shearpars_to_mompars(pars):
    e1, e2, T = pars[2:5]
    irr, irc, icc = _e2mom(e1, e2, T)
    return jnp.array([pars[0], pars[1], irr, irc, icc, pars[5]])


def _mompars_to_shearpars(pars):
    irr, irc, icc = pars[2:5]
    e1, e2, T = _mom2e(irr, irc, icc)
    return jnp.array([pars[0], pars[1], e1, e2, T, pars[5]])


def _eval_gauss2d(pars, u, v, area):

    cen_v, cen_u, irr, irc, icc, flux = pars[0:6]

    det = irr * icc - irc * irc
    idet = 1.0 / det
    drr = irr * idet
    drc = irc * idet
    dcc = icc * idet
    norm = 1.0 / (2 * jnp.pi * jnp.sqrt(det))

    # v->row, u->col in gauss
    vdiff = v - cen_v
    udiff = u - cen_u
    chi2 = (
        dcc * vdiff * vdiff
        + drr * udiff * udiff
        - 2.0 * drc * vdiff * udiff
    )

    return norm * jnp.exp(-0.5 * chi2) * area, norm


def _deweight_moments(wrr, wrc, wcc, irr, irc, icc, flags):
    detm = irr * icc - irc * irc
    flags = jax.lax.cond(
        detm <= LOW_DET_VAL,
        lambda x: x | 2**4,
        lambda x: x,
        flags,
    )

    idetm = 1.0/detm

    detw = wrr * wcc - wrc * wrc
    flags = jax.lax.cond(
        detw <= LOW_DET_VAL,
        lambda x: x | 2**4,
        lambda x: x,
        flags,
    )

    idetw = 1.0/detw

    # Nrr etc. are actually of the inverted covariance matrix
    nrr = icc * idetm - wcc * idetw
    ncc = irr * idetm - wrr * idetw
    nrc = -irc * idetm + wrc * idetw
    detn = nrr * ncc - nrc * nrc
    flags = jax.lax.cond(
        detn <= LOW_DET_VAL,
        lambda x: x | 2**4,
        lambda x: x,
        flags,
    )

    # now set from the inverted matrix
    idetn = 1.0 / detn
    irr_dw = ncc * idetn
    icc_dw = nrr * idetn
    irc_dw = -nrc * idetn
    return irr_dw, irc_dw, icc_dw, flags


def _compute_moments(mompars, u, v, area, image):
    wt_noimage, wt_norm = _eval_gauss2d(mompars, u, v, area)
    wt = wt_noimage * image
    wt_sum = jnp.sum(wt)
    cen_v = jnp.sum(wt * v) / wt_sum
    cen_u = jnp.sum(wt * u) / wt_sum
    dv = v - cen_v
    du = u - cen_u
    irr = jnp.sum(wt * dv * dv) / wt_sum
    irc = jnp.sum(wt * du * dv) / wt_sum
    icc = jnp.sum(wt * du * du) / wt_sum

    return cen_v, cen_u, irr, irc, icc, wt_sum / (jnp.sum(wt_noimage) * wt_norm * area)


@partial(jax.jit, static_argnames=("cenonly",))
def _admom_kern(i, args, cenonly):
    (
        mompars, u, v, area, image,
        etol, ttol, maxshift, e1e2T, orig_cen_u, orig_cen_v, flags,
        psf_T, converged,
    ) = args

    mompars = mompars.at[2].add(psf_T / 2.0)
    mompars = mompars.at[4].add(psf_T / 2.0)

    # set flags if det is too low
    curr_det = mompars[2] * mompars[4] - mompars[3] * mompars[3]
    flags = jax.lax.cond(
        curr_det <= LOW_DET_VAL,
        lambda x: x | 2**4,
        lambda x: x,
        flags,
    )

    cen_v, cen_u, irr, irc, icc, flux = _compute_moments(mompars, u, v, area, image)
    new_e1e2T = _mom2e(irr, irc, icc)

    # set flags is flux is non-positive
    flags = jax.lax.cond(
        flux <= 0.0,
        lambda x: x | 2**2,
        lambda x: x,
        flags,
    )

    # set flags if centroid is too far from original guess
    flags = jax.lax.cond(
        (jnp.abs(cen_u - orig_cen_u) > maxshift) | (jnp.abs(cen_v - orig_cen_v) > maxshift),
        lambda x: x | 2**1,
        lambda x: x,
        flags,
    )

    # set flags if new T is non-positive
    flags = jax.lax.cond(
        new_e1e2T[2] <= 0.0,
        lambda x: x | 2**3,
        lambda x: x,
        flags,
    )

    if not cenonly:
        irr, irc, icc, flags = _deweight_moments(mompars[2], mompars[3], mompars[4], irr, irc, icc, flags)
        mompars = mompars.at[2].add(-psf_T / 2.0)
        mompars = mompars.at[4].add(-psf_T / 2.0)
        irr = irr - psf_T / 2.0
        icc = icc - psf_T / 2.0
        new_mompars = jnp.array([
            cen_v,
            cen_u,
            irr,
            irc,
            icc,
            flux,
        ])
    else:
        mompars = mompars.at[2].add(-psf_T / 2.0)
        mompars = mompars.at[4].add(-psf_T / 2.0)
        new_mompars = jnp.array([
            cen_v,
            cen_u,
            mompars[2],
            mompars[3],
            mompars[4],
            flux,
        ])

    new_converged = jax.numpy.logical_and(
        jnp.abs(new_e1e2T[2] / e1e2T[2] - 1.0) <= ttol,
        jnp.logical_and(
            jnp.abs(new_e1e2T[0] - e1e2T[0]) <= etol,
            jnp.abs(new_e1e2T[1] - e1e2T[1]) <= etol,
        ),
    )
    cond = jax.numpy.logical_or(
        new_converged,
        jax.numpy.logical_or(
            flags != 0, converged
        ),
    )
    curr_mompars = jax.lax.cond(
        cond,
        lambda old, new: old,
        lambda old, new: new,
        mompars,
        new_mompars,
    )
    last_e1e2T = jax.lax.cond(
        cond,
        lambda old, new: old,
        lambda old, new: new,
        e1e2T,
        new_e1e2T,
    )
    converged = jnp.logical_or(converged, new_converged)

    return (
        curr_mompars, u, v, area, image,
        etol, ttol, maxshift, last_e1e2T, orig_cen_u, orig_cen_v, flags,
        psf_T, converged,
    )


def gen_guess_admom(rng_key, guess_T=0, jac_scale=1, rng_scale=1.0):
    guess_T = jnp.maximum(_fwhm_to_T(jac_scale * 5), guess_T)
    pars = jrandom.uniform(rng_key, shape=(6,), minval=-1, maxval=1)
    pars = pars * jnp.array([0.5 * jac_scale, 0.5 * jac_scale, 0.3, 0.3, 1.0, 1.0])
    pars = pars * rng_scale
    pars = pars.at[4].set(guess_T * (1.0 + pars[4]/10.0))
    pars = pars.at[5].set(1.0)
    return pars


@partial(jax.jit, static_argnames=("unroll", "cenonly", "maxitr"))
def admom(
    image,
    weight,
    cen_x, cen_y,
    dudx, dudy, dvdx, dvdy,
    guess,
    psf_T,
    maxitr=200,
    etol=1e-5,
    ttol=1e-5,
    maxshift=5.0,
    unroll=1,
    cenonly=False,
):
    # generate pixel locations and flatten to 1D
    # for easier indexing into sums
    x, y = jnp.meshgrid(
        jnp.arange(image.shape[1], dtype=float),
        jnp.arange(image.shape[0], dtype=float),
    )
    x = x.flatten()
    y = y.flatten()
    dx = x - cen_x
    dy = y - cen_y
    u = dudx * dx + dudy * dy
    v = dvdx * dx + dvdy * dy
    image = image.flatten()
    weight = weight.flatten()

    scale = _get_jac_scale(dudx, dudy, dvdx, dvdy)
    area = scale * scale

    mompars = _shearpars_to_mompars(guess)
    init_e1e2T = (1e12, 1e12, 1e12)

    # loop over iterations
    res = (
        mompars, u, v, area, image, etol, ttol, maxshift,
        init_e1e2T, mompars[0], mompars[1], 0, psf_T, False,
    )
    res = jax.lax.fori_loop(
        0,
        maxitr,
        partial(_admom_kern, cenonly=cenonly),
        res,
        unroll=unroll,
    )
    flags = res[-3]

    flags = jax.lax.cond(
        res[-1],
        lambda x: x,
        lambda x: x | 2**5,
        flags,
    )

    return _mompars_to_shearpars(res[0]), flags
