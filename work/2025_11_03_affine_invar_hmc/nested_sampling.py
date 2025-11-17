import tqdm
from typing import NamedTuple

import jax.numpy
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jrng


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

    min_loglike: float
    """
    Current minimum log-likelihood. Initial value is -inf.
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

    rng_key: jax.numpy.ndarray
    """The current jax RNG key."""


def nested_sampler(
    rng_key,
    log_likelihood,
    prior_draw,
    constrained_sampler,
    n_dims,
    n_live,
    n_iter_max,
    n_iter_conv_fac,
):
    """Run nested sampling.

    Parameters
    ----------
    rng_key : PRNG key
        The RNG key to use.
    log_likelihood : callable
        A callable with signature `log_likelihood(x)`
        that returns the likelihood for a single sample point.
    prior_draw : callable
        A callable with signature `prior_draw(rng_key)` that draws a
        point from the prior.
    constrained_sampler : callable
        A callable with signature `constrained_sampler(rng_key, ns_data, dead_index)`
        that returns a new sample from the prior with the constraint that
        `log_likelihood(sample) > ns_data.min_loglike`.
    n_dims : int
        The number of dimensions.
    n_live : int
        The number of live points.
    n_iter_max : int
        The maximum number of iterations.
    n_iter_conv_fac : float
        The convergence factor defined so that convergence happens when
        `n_iter > n_iter_conv_fac * n_iter * H`. A typical value is 2.
    
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

    new_keys = jrng.split(rng_key, num=n_live + 1)
    rng_key = new_keys[0]
    theta_keys = new_keys[1:]
    theta = jax.vmap(prior_draw)(theta_keys)

    live_points = NSPointSet(
        theta=theta,
        loglike=jax.vmap(log_likelihood)(theta),
        logwlike=jnp.zeros((n_live)) - jnp.inf,
    )

    sample_points = NSPointSet(
        theta=jnp.zeros((n_iter_max, n_dims)),
        loglike=jnp.zeros(n_iter_max),
        logwlike=jnp.zeros(n_iter_max),
    )

    ns_data = NSData(
        n_live=n_live,
        n_iter_max=n_iter_max,
        n_iter=0,
        logw=jnp.log(1.0 - jnp.exp(-1.0 / n_live)),
        min_loglike=-jnp.inf,
        H=0.0,
        logZ=-jnp.inf,
        n_iter_conv_fac=n_iter_conv_fac,
        live_points=live_points,
        rng_key=rng_key,
    )

    with tqdm.trange(n_iter_max, ncols=80, desc="sampling") as pbar:
        for _ in pbar:
            # find the current dead point
            dead_index = jnp.argmin(ns_data.live_points.loglike)
            dead_loglike = ns_data.live_points.loglike[dead_index]
            dead_logwlike = ns_data.logw + dead_loglike
            dead_theta = ns_data.live_points.theta[dead_index, :]

            # add dead point to samples
            sample_points = NSPointSet(
                theta=sample_points.theta.at[ns_data.n_iter, :].set(dead_theta),
                loglike=sample_points.loglike.at[ns_data.n_iter].set(dead_loglike),
                logwlike=sample_points.logwlike.at[ns_data.n_iter].set(dead_logwlike),
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
            rng_key, sampler_key = jrng.split(rng_key)
            new_point = constrained_sampler(sampler_key, ns_data, dead_index)
            live_points = NSPointSet(
                theta=ns_data.live_points.theta.at[dead_index, :].set(new_point),
                loglike=ns_data.live_points.loglike.at[dead_index].set(log_likelihood(new_point)),
                logwlike=None,
            )

            # build new ns_data structure
            ns_data = NSData(
                n_live=n_live,
                n_iter_max=n_iter_max,
                n_iter=ns_data.n_iter + 1,
                logw=new_logw,
                min_loglike=dead_loglike,
                H=new_H,
                logZ=new_logZ,
                n_iter_conv_fac=n_iter_conv_fac,
                live_points=live_points,
                rng_key=rng_key,
            )

            new_conv_total = jnp.int_(ns_data.n_iter_conv_fac * ns_data.n_live * ns_data.H)

            if ns_data.n_iter % 100 == 0:
                pbar.total = int(new_conv_total)
                pbar.refresh()
                pbar.set_description(
                    "sampling (logZ: %.5e)" % (
                        new_logZ,
                    )
                )

            if (
                # convergence
                (ns_data.n_iter > new_conv_total)
                # we do not have enough storage room for correction points
                or (ns_data.n_iter_max - ns_data.n_iter < ns_data.n_live)
            ):
                pbar.update()
                pbar.total = int(ns_data.n_iter)
                pbar.set_description(
                    "sampling (logZ: %.5e)" % (
                        new_logZ,
                    )
                )
                pbar.refresh()
                break

    # compute correction
    def _accumulate_correction(carry, dead_index):
        _final_logZ, _final_H, _logw, _ns_data, _sample_points = carry

        dead_loglike = _ns_data.live_points.loglike[dead_index]
        dead_logwlike = _logw + dead_loglike
        dead_theta = _ns_data.live_points.theta[dead_index, :]

        # update evidence, H, and logw
        new_final_logZ = jsp.special.logsumexp(jnp.array([_final_logZ, dead_logwlike]))
        _final_H = (
            jnp.exp(dead_logwlike - new_final_logZ) * dead_loglike
            + jnp.exp(final_logZ - new_final_logZ) * (final_H + final_logZ) - new_final_logZ
        )
        _final_logZ = new_final_logZ

        # add dead point to samples
        _sample_points = NSPointSet(
            theta=_sample_points.theta.at[_ns_data.n_iter + dead_index].set(dead_theta),
            loglike=_sample_points.loglike.at[_ns_data.n_iter + dead_index].set(dead_loglike),
            logwlike=_sample_points.logwlike.at[_ns_data.n_iter + dead_index].set(dead_logwlike),
        )

        return (
            (
                _final_logZ,
                _final_H,
                _logw,
                _ns_data,
                _sample_points
            ),
            None,
        )

    logw = -ns_data.n_iter / ns_data.n_live - jnp.log(ns_data.n_live)
    final_logZ = ns_data.logZ
    final_H = ns_data.H
    res, _ = jax.lax.scan(
        _accumulate_correction,
        (
            final_logZ,
            final_H,
            logw,
            ns_data,
            sample_points
        ),
        xs=jnp.arange(ns_data.n_live)
    )
    (
        final_logZ,
        final_H,
        _,
        ns_data,
        sample_points
    ) = res

    samples = sample_points.theta[:ns_data.n_iter + ns_data.n_live, :]
    log_like = sample_points.loglike[:ns_data.n_iter + ns_data.n_live]
    log_weights = sample_points.logwlike[:ns_data.n_iter + ns_data.n_live]
    log_weights = log_weights - jsp.special.logsumexp(log_weights)

    return (
        final_logZ,
        jnp.sqrt(final_H / n_live),
        samples,
        log_weights,
        log_like,
        ns_data,
    )
