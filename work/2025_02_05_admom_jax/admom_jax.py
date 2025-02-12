from functools import partial

import jax
import jax.numpy as jnp


from admom_core import (
    compute_moments_admom_obs,
    deweight_moments,
    mom2e,
    mompars_to_shearpars,
    obs_to_admom_obs,
    shearpars_to_mompars,
    AdMomData,
    Obs,
    LOW_DET_VAL
)


@partial(jax.jit, static_argnames=("cenonly",))
def _admom_kern_inner_loop(mompars, obs, flags, admom_data, cenonly):
    mompars = mompars.at[2].add(obs.psf_T / 2.0)
    mompars = mompars.at[4].add(obs.psf_T / 2.0)

    # set flags if det is too low
    curr_det = mompars[2] * mompars[4] - mompars[3] * mompars[3]
    flags = jax.lax.cond(
        curr_det <= LOW_DET_VAL,
        lambda x: x | 2**4,
        lambda x: x,
        admom_data.flags,
    )

    cen_v, cen_u, raw_irr, raw_irc, raw_icc, flux = compute_moments_admom_obs(mompars, obs)

    # set flags is flux is non-positive
    flags = jax.lax.cond(
        flux <= 0.0,
        lambda x: x | 2**2,
        lambda x: x,
        flags,
    )

    if not cenonly:
        irr, irc, icc, flags = deweight_moments(
            mompars[2], mompars[3], mompars[4], raw_irr, raw_irc, raw_icc, flags
        )
        mompars = mompars.at[2].add(-obs.psf_T / 2.0)
        mompars = mompars.at[4].add(-obs.psf_T / 2.0)
        irr = irr - obs.psf_T / 2.0
        icc = icc - obs.psf_T / 2.0
    else:
        mompars = mompars.at[2].add(-obs.psf_T / 2.0)
        mompars = mompars.at[4].add(-obs.psf_T / 2.0)
        irr = raw_irr
        irc = raw_irc
        icc = raw_icc

    obs_moms = jnp.array([
        cen_v,
        cen_u,
        irr,
        irc,
        icc,
        flux,
    ])

    return obs_moms, flags, raw_irr, raw_irc, raw_icc


@partial(jax.jit, static_argnames=("cenonly", "n_obs"))
def _admom_kern(i, args, cenonly, n_obs):
    mompars, admom_data = args
    flags = admom_data.flags

    tot_moms = jnp.zeros(5)
    fluxes = jnp.zeros(n_obs)
    tot_wgt = 0.0
    irr = 0.0
    irc = 0.0
    icc = 0.0
    for i in range(n_obs):
        obs = admom_data.obs[i]
        obs_moms, _flags, _irr, _irc, _icc = _admom_kern_inner_loop(mompars, obs, flags, admom_data, cenonly)
        tot_moms = tot_moms + obs_moms[0:5] * obs.wgt
        tot_wgt = tot_wgt + obs.wgt
        fluxes = fluxes.at[i].set(obs_moms[5])
        flags = flags | _flags
        irr = irr + _irr * obs.wgt
        irc = irc + _irc * obs.wgt
        icc = icc + _icc * obs.wgt

    new_mompars = tot_moms / tot_wgt
    if cenonly:
        new_mompars = new_mompars.at[2].set(mompars[2])
        new_mompars = new_mompars.at[3].set(mompars[3])
        new_mompars = new_mompars.at[4].set(mompars[4])
    new_mompars = jnp.concat([new_mompars, fluxes])

    irr = irr / tot_wgt
    irc = irc / tot_wgt
    icc = icc / tot_wgt
    new_e1e2T = mom2e(irr, irc, icc)

    # set flags if centroid is too far from original guess
    flags = jax.lax.cond(
        (
            (jnp.abs(new_mompars[0] - admom_data.orig_cen_u) > admom_data.maxshift)
            | (jnp.abs(new_mompars[1] - admom_data.orig_cen_v) > admom_data.maxshift)
        ),
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

    new_converged = jax.numpy.logical_and(
        jnp.abs(new_e1e2T[2] / admom_data.e1e2T[2] - 1.0) <= admom_data.ttol,
        jnp.logical_and(
            jnp.abs(new_e1e2T[0] - admom_data.e1e2T[0]) <= admom_data.etol,
            jnp.abs(new_e1e2T[1] - admom_data.e1e2T[1]) <= admom_data.etol,
        ),
    )
    cond = jax.numpy.logical_or(
        new_converged,
        jax.numpy.logical_or(
            flags != 0, admom_data.converged
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
        admom_data.e1e2T,
        new_e1e2T,
    )
    converged = jnp.logical_or(admom_data.converged, new_converged)

    return curr_mompars, AdMomData(
        etol=admom_data.etol,
        ttol=admom_data.ttol,
        maxshift=admom_data.maxshift,
        obs=admom_data.obs,
        flags=flags,
        converged=converged,
        e1e2T=last_e1e2T,
        orig_cen_u=admom_data.orig_cen_u,
        orig_cen_v=admom_data.orig_cen_v,
    )


@partial(jax.jit, static_argnames=("unroll", "cenonly", "maxitr"))
def admom(
    obs: list[Obs],
    guess,
    maxitr=200,
    etol=1e-5,
    ttol=1e-5,
    maxshift=5.0,
    unroll=1,
    cenonly=False,
):
    mompars = shearpars_to_mompars(guess)

    admom_data = AdMomData(
        etol=etol,
        ttol=ttol,
        maxshift=maxshift,
        obs=[obs_to_admom_obs(_obs) for _obs in obs],
        flags=0,
        converged=False,
        e1e2T=(1e12, 1e12, 1e12),
        orig_cen_u=mompars[1],
        orig_cen_v=mompars[0],
    )

    # loop over iterations
    res = jax.lax.fori_loop(
        0,
        maxitr,
        partial(_admom_kern, cenonly=cenonly, n_obs=len(admom_data.obs)),
        (mompars, admom_data),
        unroll=unroll,
    )
    flags = res[1].flags
    flags = jax.lax.cond(
        res[1].converged,
        lambda x: x,
        lambda x: x | 2**5,
        flags,
    )

    return mompars_to_shearpars(res[0]), flags
