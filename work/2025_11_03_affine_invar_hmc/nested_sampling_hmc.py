import functools
import tqdm
from typing import NamedTuple

import jax
import jax.numpy
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jrng


from affine_invar_constr_hmc import ensemble_hmc_with_constraint


class NSPointSet(NamedTuple):
    """Nested sampling point set.
    
    This data structure holds both the current set of live
    points and the running list of dead points.
    """

    theta: jax.numpy.ndarray
    """The parameters values for each point in the point
    set. The shape is (n_points, n_dim).
    """

    loglike: jax.numpy.ndarray
    """
    The log-likelihood associated with each point. The
    shape is (n_points,).
    """

    logwlike: jax.numpy.ndarray
    """
    The value log(w * likelihood) such that the evidence is
    `logZ = jax.scipy.special.logsumexp(wL)`. The
    shape is (n_points,).
    """


class NSData(NamedTuple):
    """Nested sampling data."""

    n_live: int
    """
    The number of live points.
    """

    n_iter_max: int
    """
    Maximum number of iterations.
    """

    n_iter: int
    """
    Current number of iterations.
    """

    logw: float
    """
    Log of the weight (volume) for the next dead point. Initial
    value is log(1 - exp(-1 / n_live)). It decreases by 1 / n_live
    on each iteration.
    """

    min_loglike_ind: int
    """
    The index of the live point with the minimum log-likelihood.
    """

    next_min_loglike_ind: int
    """
    The index of the live point with the next minimum log-likelihood.
    """

    H: float
    """
    Current estimate of KL divergence between posterior and prior. Initial
    value is 0.0.
    """

    logZ: float
    """
    Current estimate of log-evidence. Initial value is -inf.
    """

    n_iter_conv_fac: float
    """
    Convergence happens when n_iter > n_iter_conv_fac * n_iter * H.
    A typical value is 10.
    """

    live_points: NSPointSet
    """
    Current set of live points.
    """

    sample_points: NSPointSet
    """
    Set of dead points forming the nested samples.
    """

    rng_key: jax.numpy.ndarray
    """The current jax RNG key."""


def _ns_data_truncate_sample_points(ns_data):
    """A helper function to truncate the sample points to their final length."""
    return NSData(
        n_live=ns_data.n_live,
        n_iter_max=ns_data.n_iter_max,
        n_iter=ns_data.n_iter,
        logw=ns_data.logw,
        min_loglike_ind=ns_data.min_loglike_ind,
        next_min_loglike_ind=ns_data.min_loglike_ind,
        H=ns_data.H,
        logZ=ns_data.logZ,
        n_iter_conv_fac=ns_data.n_iter_conv_fac,
        live_points=ns_data.live_points,
        sample_points=NSPointSet(
            theta=ns_data.sample_points.theta[:ns_data.n_iter + ns_data.n_live, :],
            loglike=ns_data.sample_points.loglike[:ns_data.n_iter + ns_data.n_live],
            logwlike=ns_data.sample_points.logwlike[:ns_data.n_iter + ns_data.n_live],
        ),
        rng_key=ns_data.rng_key,
    )


def _ns_data_set_logw(ns_data, logw):
    """A helper function to truncate the sample points to their final length."""
    return NSData(
        n_live=ns_data.n_live,
        n_iter_max=ns_data.n_iter_max,
        n_iter=ns_data.n_iter,
        logw=logw,
        min_loglike_ind=ns_data.min_loglike_ind,
        next_min_loglike_ind=ns_data.min_loglike_ind,
        H=ns_data.H,
        logZ=ns_data.logZ,
        n_iter_conv_fac=ns_data.n_iter_conv_fac,
        live_points=ns_data.live_points,
        sample_points=ns_data.sample_points,
        rng_key=ns_data.rng_key,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "n_live",
        "log_likelihood",
        "log_prior",
        "n_dims",
        "n_walkers_hmc",
        "n_samples_hmc",
        "leapfrog_step_size_hmc",
        "n_leapfrog_steps_hmc",
    )
)
def _constrained_sampler(
    sampler_key,
    live_points_theta,
    n_live,
    dead_index,
    dead_loglike,
    log_likelihood,
    log_prior,
    n_dims,
    n_walkers_hmc,
    n_samples_hmc,
    leapfrog_step_size_hmc,
    n_leapfrog_steps_hmc,
):

    def _constraint(x):
        return jnp.exp(log_likelihood(x) - dead_loglike) - 1.0

    def _neg_log_like(x):
        return -1.0 * log_prior(x)

    sampler_key, draw_key = jrng.split(sampler_key)
    draw_index = jrng.choice(draw_key, n_live)
    live_points_theta = live_points_theta.at[dead_index, :].set(live_points_theta[draw_index, :])

    sampler_key, draw_key = jrng.split(sampler_key)
    draw_index = jrng.choice(draw_key, n_live, shape=(n_walkers_hmc,), replace=False)
    live_points_theta = live_points_theta[draw_index, :]

    sampler_key, hmc_key = jrng.split(sampler_key)
    chain, _, _ = ensemble_hmc_with_constraint(
        _neg_log_like,
        live_points_theta,
        n_samples_hmc,
        n_dims,
        n_walkers_hmc,
        leapfrog_step_size_hmc,
        n_leapfrog_steps_hmc,
        _constraint,
        hmc_key,
        verbose=False,
    )

    sampler_key, draw_key = jrng.split(sampler_key)
    draw_index = jrng.choice(draw_key, n_walkers_hmc)
    return chain[-1, draw_index, :]


def _sort2(m1, m2, arr):
    return jax.lax.cond(
        arr[m1] < arr[m2],
        lambda x, y: (x, y),
        lambda x, y: (y, x),
        m1,
        m2,
    )

def _min_two(arr):
    m1 = 0
    m2 = 1
    m1, m2 = _sort2(m1, m2, arr)

    def _min_op(carry, ind):
        _m1, _m2 = carry

        _m1, _m2, _ = jax.lax.cond(
            arr[ind] < arr[_m1],
            lambda x, y, z: (z, x, y),
            lambda x, y, z: jax.lax.cond(
                arr[ind] < arr[_m2],
                lambda _x, _y, _z: (_x, _z, _y),
                lambda _x, _y, _z: (_x, _y, _z),
                x,
                y,
                z,
            ),
            _m1,
            _m2,
            ind,
        )

        return (_m1, _m2), None

    return jax.lax.scan(
        _min_op,
        (m1, m2),
        xs=jnp.arange(arr.shape[0])
    )[0]


def nested_sampler_hmc(
    rng_key,
    log_likelihood,
    log_prior,
    prior_draw,
    n_dims,
    n_live,
    n_iter_max,
    n_iter_conv_fac,
    n_walkers_hmc,
    n_samples_hmc,
    leapfrog_step_size_hmc,
    n_leapfrog_steps_hmc,
    n_conv_check=10,
):
    """Run nested sampling.

    Parameters
    ----------
    rng_key : PRNG key
        The RNG key to use.
    log_likelihood : callable
        A callable with signature `log_likelihood(x)`
        that returns the log-likelihood for a single sample point.
    log_prior : callable
        A callable with signature `log_prior(x)` that returns the
        log-prior for a single point.
    prior_draw : callable
        A callable with signature `prior_draw(rng_key)` that draws a
        point from the prior.
    n_dims : int
        The number of dimensions.
    n_live : int
        The number of live points.
    n_iter_max : int
        The maximum number of iterations.
    n_iter_conv_fac : float
        The convergence factor defined so that convergence happens when
        `n_iter > n_iter_conv_fac * n_iter * H`. A typical value is 2.
    n_walkers_hmc : int
        The number of walkers to use for HMC.
    n_samples_hmc : int
        The number of samples to produce when drawing the new live point.
    leapfrog_step_size_hmc : float
        The step size for the leapfrog integration.
    n_leapfrog_steps_hmc : int
        The number of leapfrog steps to take.
    n_conv_check : int, optional
        How often to check convergence. Default is every 10 iterations.
    
    Returns
    -------
    log_evidence : float
        The estimated evidence.
    delta_log_devidence : float
        The estimated remaining evidence after the algorithm terminates. This
        quantity is **already added to the `log_evidence`**.  It can be used as an 
        estimate of the error in the `log_evidence`.
    samples : jax.numpy.ndarray
        The array of dead points, shape (n_iter, n_dims).
    log_weights : jax.numpy.ndarray
        The log(weight) for each sample, shape (n_iter).
    log_like : jax.numpy.ndarray
        The log-likelihood for each sample, shape (n_iter).
    ns_data : NSData
        The final nested sampling data holding the internal state of the
        sampler when it terminates.
    """

    assert n_walkers_hmc <= n_live or n_walkers_hmc >= 2 * n_dims, (
        "The parameter `n_walkers_hmc must satisfy "
        "`2 * n_dims <= n_walkers_hmc <= n_live`! "
        f"You sent n_walkers_hmc={n_walkers_hmc}, n_dims={n_dims}, n_live={n_live}."
    )

    def _nested_sampling_itr(carry, itr):
        ns_data = carry

        # find the current dead point
        dead_index = ns_data.min_loglike_ind
        dead_loglike = ns_data.live_points.loglike[dead_index]
        dead_logwlike = ns_data.logw + dead_loglike
        dead_theta = ns_data.live_points.theta[dead_index, :]

        # add dead point to samples
        sample_points = NSPointSet(
            theta=ns_data.sample_points.theta.at[ns_data.n_iter, :].set(dead_theta),
            loglike=ns_data.sample_points.loglike.at[ns_data.n_iter].set(dead_loglike),
            logwlike=ns_data.sample_points.logwlike.at[ns_data.n_iter].set(dead_logwlike),
        )

        # update evidence, H, and logw
        new_logZ = jsp.special.logsumexp(jnp.array([ns_data.logZ, dead_logwlike]))
        new_H = (
            jnp.exp(dead_logwlike - new_logZ) * dead_loglike
            + jnp.where(
                ns_data.logZ == -jnp.inf,
                0.0,
                jnp.exp(ns_data.logZ - new_logZ) * (ns_data.H + ns_data.logZ),
            ) - new_logZ
        )
        new_logw = ns_data.logw - 1.0 / ns_data.n_live

        # replace dead point with something else in the live point set
        _rng_key, sampler_key = jrng.split(ns_data.rng_key)
        new_point = _constrained_sampler(
            sampler_key,
            ns_data.live_points.theta,
            n_live,
            dead_index,
            dead_loglike,
            log_likelihood,
            log_prior,
            n_dims,
            n_walkers_hmc,
            n_samples_hmc,
            leapfrog_step_size_hmc,
            n_leapfrog_steps_hmc,
        )
        live_points = NSPointSet(
            theta=ns_data.live_points.theta.at[dead_index, :].set(new_point),
            loglike=ns_data.live_points.loglike.at[dead_index].set(log_likelihood(new_point)),
            logwlike=None,
        )

        min_loglike_ind = dead_index
        next_min_loglike_ind = ns_data.next_min_loglike_ind
        min_loglike_ind, next_min_loglike_ind = jax.lax.cond(
            live_points.loglike[min_loglike_ind] < live_points.loglike[next_min_loglike_ind],
            lambda x0, x1: (x0, x1),
            lambda x0, x1: _min_two(live_points.loglike),
            min_loglike_ind,
            next_min_loglike_ind
        )

        # build new ns_data structure
        ns_data = NSData(
            n_live=ns_data.n_live,
            n_iter_max=ns_data.n_iter_max,
            n_iter=ns_data.n_iter + 1,
            logw=new_logw,
            min_loglike_ind=min_loglike_ind,
            next_min_loglike_ind=next_min_loglike_ind,
            H=new_H,
            logZ=new_logZ,
            n_iter_conv_fac=ns_data.n_iter_conv_fac,
            live_points=live_points,
            sample_points=sample_points,
            rng_key=_rng_key,
        )

        return ns_data, None

    # compute correction
    def _accumulate_correction(carry, dead_index):
        ns_data = carry

        dead_loglike = ns_data.live_points.loglike[dead_index]
        dead_logwlike = ns_data.logw + dead_loglike
        dead_theta = ns_data.live_points.theta[dead_index, :]

        # update evidence, H, and logw
        new_logZ = jsp.special.logsumexp(jnp.array([ns_data.logZ, dead_logwlike]))
        new_H = (
            jnp.exp(dead_logwlike - new_logZ) * dead_loglike
            + jnp.exp(ns_data.logZ - new_logZ) * (ns_data.H + ns_data.logZ) - new_logZ
        )

        # build new ns_data structure
        ns_data = NSData(
            n_live=ns_data.n_live,
            n_iter_max=ns_data.n_iter_max,
            n_iter=ns_data.n_iter,
            logw=ns_data.logw,
            min_loglike_ind=ns_data.min_loglike_ind,
            next_min_loglike_ind=ns_data.min_loglike_ind,
            H=new_H,
            logZ=new_logZ,
            n_iter_conv_fac=n_iter_conv_fac,
            live_points=ns_data.live_points,
            sample_points=NSPointSet(
                theta=ns_data.sample_points.theta.at[ns_data.n_iter + dead_index].set(dead_theta),
                loglike=ns_data.sample_points.loglike.at[ns_data.n_iter + dead_index].set(dead_loglike),
                logwlike=ns_data.sample_points.logwlike.at[ns_data.n_iter + dead_index].set(dead_logwlike),
            ),
            rng_key=ns_data.rng_key,
        )

        return ns_data, None

    new_keys = jrng.split(rng_key, num=n_live + 1)
    rng_key = new_keys[0]
    theta_keys = new_keys[1:]
    theta = jax.vmap(prior_draw)(theta_keys)
    _live_points = NSPointSet(
        theta=theta,
        loglike=jax.vmap(log_likelihood)(theta),
        logwlike=None,
    )
    _sorted_inds = jnp.argsort(_live_points.loglike)[0:2]
    ns_data = NSData(
        n_live=n_live,
        n_iter_max=n_iter_max,
        n_iter=0,
        logw=jnp.log(1.0 - jnp.exp(-1.0 / n_live)),
        min_loglike_ind=_sorted_inds[0],
        next_min_loglike_ind=_sorted_inds[1],
        H=0.0,
        logZ=-jnp.inf,
        n_iter_conv_fac=n_iter_conv_fac,
        live_points=_live_points,
        sample_points=NSPointSet(
            theta=jnp.zeros((n_iter_max, n_dims)),
            loglike=jnp.zeros(n_iter_max),
            logwlike=jnp.zeros(n_iter_max),
        ),
        rng_key=rng_key,
    )

    curr_iter = 0
    n_scan_calls = n_iter_max // n_conv_check
    if n_scan_calls * n_conv_check < n_iter_max:
        n_scan_calls += 1
    with tqdm.trange(n_iter_max, ncols=80, desc="sampling") as pbar:
        for _ in range(n_scan_calls):
            if n_conv_check + curr_iter > n_iter_max:
                _n_to_do = n_iter_max - curr_iter
            else:
                _n_to_do = n_conv_check

            # we won't have enough room for the live points after this iteration
            # so we break
            if (ns_data.n_iter_max - ns_data.n_iter < ns_data.n_live + _n_to_do):
                pbar.total = int(ns_data.n_iter)
                pbar.refresh()
                break

            ns_data, _ = jax.lax.scan(
                _nested_sampling_itr,
                ns_data,
                length=_n_to_do,
            )
            pbar.update(_n_to_do)
            curr_iter += _n_to_do

            new_conv_total = jnp.int_(ns_data.n_iter_conv_fac * ns_data.n_live * ns_data.H)

            pbar.total = int(new_conv_total)
            pbar.set_description(
                "sampling (logZ: %.5e)" % (
                    ns_data.logZ,
                )
            )
            pbar.refresh()

            # convergence
            if ns_data.n_iter > new_conv_total:
                pbar.total = int(ns_data.n_iter)
                pbar.refresh()
                break

    # truncate the sample points to length n_iter + n_live
    ns_data = _ns_data_truncate_sample_points(ns_data)
    # set the final logw value
    ns_data = _ns_data_set_logw(ns_data, -ns_data.n_iter / ns_data.n_live - jnp.log(ns_data.n_live))
    ns_data, _ = jax.lax.scan(
        _accumulate_correction,
        ns_data,
        xs=jnp.arange(ns_data.n_live)
    )

    samples = ns_data.sample_points.theta
    log_like = ns_data.sample_points.loglike
    log_weights = ns_data.sample_points.logwlike
    log_weights = log_weights - jsp.special.logsumexp(log_weights)

    return (
        ns_data.logZ,
        jnp.sqrt(ns_data.H / n_live),
        samples,
        log_weights,
        log_like,
        ns_data,
    )
