from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrandom

LOW_DET_VAL = 1.0e-200


@jax.tree_util.register_pytree_node_class
class AdMomObs(NamedTuple):
    u: jax.Array
    v: jax.Array
    image: jax.Array
    area: float
    psf_T: float
    wgt: float

    def tree_flatten(self):
        return (self.u, self.v, self.image, self.area, self.psf_T, self.wgt), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class AdMomData(NamedTuple):
    ttol: float
    etol: float
    maxshift: float
    obs: list[AdMomObs]
    flags: int
    converged: bool
    e1e2T: tuple[float]
    orig_cen_u: float
    orig_cen_v: float

    def tree_flatten(self):
        return (self.ttol, self.etol, self.maxshift, self.obs, self.flags, self.converged, self.e1e2T, self.orig_cen_u, self.orig_cen_v), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class Obs(NamedTuple):
    image: jax.Array
    weight: jax.Array
    cen_x: float
    cen_y: float
    dudx: float
    dudy: float
    dvdx: float
    dvdy: float
    psf_T: float | None = None

    def tree_flatten(self):
        return (self.image, self.weight, self.cen_x, self.cen_y, self.dudx, self.dudy, self.dvdx, self.dvdy, self.psf_T), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def obs_to_admom_obs(obs: Obs) -> AdMomObs:
    x, y = jnp.meshgrid(
        jnp.arange(obs.image.shape[1], dtype=float),
        jnp.arange(obs.image.shape[0], dtype=float),
    )
    dx = x - obs.cen_x
    dy = y - obs.cen_y
    u = obs.dudx * dx + obs.dudy * dy
    v = obs.dvdx * dx + obs.dvdy * dy

    return AdMomObs(
        u,
        v,
        obs.image,
        obs.dudx * obs.dvdy - obs.dudy * obs.dvdx,
        obs.psf_T,
        jnp.median(obs.weight),
    )


def fwhm_to_T(fwhm):
    sigma = fwhm / 2.3548200450309493
    return 2 * sigma ** 2


def e2mom(e1, e2, T):
    T_2 = 0.5 * T
    irc = e2 * T_2
    icc = (1 + e1) * T_2
    irr = (1 - e1) * T_2

    return irr, irc, icc


def mom2e(irr, irc, icc):
    T = irr + icc
    e1 = (icc - irr) / T
    e2 = 2.0 * irc / T

    return e1, e2, T


def shearpars_to_mompars(pars):
    e1, e2, T = pars[2:5]
    irr, irc, icc = e2mom(e1, e2, T)
    return jnp.concatenate([
        jnp.array([pars[0], pars[1], irr, irc, icc]),
        pars[5:]
    ])


def mompars_to_shearpars(pars):
    irr, irc, icc = pars[2:5]
    e1, e2, T = mom2e(irr, irc, icc)
    return jnp.concatenate([
        jnp.array([pars[0], pars[1], e1, e2, T]),
        pars[5:]
    ])


def _eval_gauss2d(pars, u, v, area):

    cen_v, cen_u, irr, irc, icc = pars[0:5]

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


def deweight_moments(wrr, wrc, wcc, irr, irc, icc, flags):
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


def compute_moments_admom_obs(mompars, admom_obs: AdMomObs):
    wt_noimage, wt_norm = _eval_gauss2d(mompars, admom_obs.u, admom_obs.v, admom_obs.area)
    wt = wt_noimage * admom_obs.image
    wt_sum = jnp.sum(wt)
    cen_v = jnp.sum(wt * admom_obs.v) / wt_sum
    cen_u = jnp.sum(wt * admom_obs.u) / wt_sum
    dv = admom_obs.v - cen_v
    du = admom_obs.u - cen_u
    irr = jnp.sum(wt * dv * dv) / wt_sum
    irc = jnp.sum(wt * du * dv) / wt_sum
    icc = jnp.sum(wt * du * du) / wt_sum

    return cen_v, cen_u, irr, irc, icc, wt_sum / (jnp.sum(wt_noimage) * wt_norm * admom_obs.area)


def gen_guess_admom(rng_key, n_obs=1, guess_T=0, jac_scale=1, rng_scale=1.0):
    guess_T = jnp.maximum(fwhm_to_T(jac_scale * 5), guess_T)
    pars = jrandom.uniform(rng_key, shape=(5 + n_obs,), minval=-1, maxval=1)
    pars = pars * jnp.concatenate([
        jnp.array([0.5 * jac_scale, 0.5 * jac_scale, 0.3, 0.3, 1.0]),
        jnp.ones(n_obs),
    ])
    pars = pars * rng_scale
    pars = pars.at[4].set(guess_T * (1.0 + pars[4]/10.0))
    for i in range(n_obs):
        pars = pars.at[5+i].set(1.0)
    return pars
