import functools

import jax
import jax.numpy as jnp
import jax.random as jrng
import numpy as np

from folx import forward_laplacian
from jax_tqdm import scan_tqdm


@functools.partial(jax.jit, static_argnames="fun")
def _value_and_grad_naive_laplacian(x, fun):
    vandg = jax.value_and_grad(fun)(x)
    vandg_lap = jax.value_and_grad(lambda x: jnp.sum(jnp.trace(jax.hessian(fun)(x))))(x)
    return vandg[0], vandg[1], vandg_lap[0], vandg_lap[1]


@functools.partial(jax.jit, static_argnames="fun")
def _value_and_grad_forward_laplacian(x, fun):
    def _fun_for_grad_flp(x):
        res = forward_laplacian(fun)(x)
        return res.laplacian, res
    grad_lp, res = jax.jacfwd(_fun_for_grad_flp, has_aux=True)(x)
    return res.x, res.jacobian.dense_array, res.laplacian, grad_lp


@functools.partial(jax.jit, static_argnames="fun")
def _parlap(x, fun):
    x = x.reshape(-1)
    n = x.shape[0]
    eye = jnp.eye(n, dtype=x.dtype)
    value_and_grad_fun = jax.value_and_grad(fun)
    def grad_and_value_fun(x):
        return value_and_grad_fun(x)[::-1]
    jacobian, dgrad_fun, val = jax.linearize(grad_and_value_fun, x, has_aux=True)

    laplacian = jnp.sum(jnp.diagonal(jax.vmap(dgrad_fun)(eye)))
    return val, jacobian, laplacian

@functools.partial(jax.jit, static_argnames="fun")
def _value_and_grad_parallel_laplacian(x, fun):
    def _fun_for_grad(x):
        res = _parlap(x, fun)
        return res[-1], res
    grad_lp, res = jax.grad(_fun_for_grad, has_aux=True)(x)
    return res[0], res[1], res[2], grad_lp


# @functools.partial(jax.jit, static_argnames=("lfun", "dt", "sqrtd", "hbar", "m", "eps"))
# def _kick(lfun, q, p, r, pr, m, hbar, dt, sqrtd, eps):
#     lval, glval, lplval, glplval = _value_and_grad_forward_laplacian(q, lfun)

#     qpot = lplval
#     gqpot = glplval

#     if hbar > 0:
#         p = p + dt * (glval + r / 2 / sqrtd * gqpot)
#     else:
#         p = p + dt * glval
#     rpe = r + eps
#     pr = pr + dt * (qpot / 2 / sqrtd - sqrtd * sqrtd * sqrtd * hbar * hbar / 8 / m / rpe / rpe)

#     return q, p, r, pr, lval, qpot


# @functools.partial(jax.jit, static_argnames=("lfun", "dt", "sqrtd", "hbar", "m", "eps"))
# def _drift(lfun, q, p, r, pr, m, hbar, dt, sqrtd, eps):
#     q = q + dt / m * p
#     pr_fac = 1.0 + dt * 2.0 / m / sqrtd * pr
#     rpe = r + eps
#     rpe = rpe * pr_fac * pr_fac
#     r = rpe - eps
#     pr = pr / pr_fac

#     return q, p, r, pr


def rpr2ab(r, pr, d, hbar):
    a = 2.0 / jnp.sqrt(d) * pr
    b = hbar * jnp.sqrt(d) / 2 / r
    return a, b


def ab2rpr(a, b, d, hbar):
    pr = jnp.sqrt(d) / 2.0 * a
    r = hbar * jnp.sqrt(d) / 2 / b
    return r, pr


@functools.partial(jax.jit, static_argnames=("vfun", "d", "hbar", "m"))
def _kick_pois(q, p, r, pr, vfun, dt, d, hbar, m):
    v, gv, lpv, glpv = _value_and_grad_forward_laplacian(q, vfun)
    if hbar > 0:
        p = p - dt * (gv + r / 2 / jnp.sqrt(d) * glpv)
    else:
        p = p - dt * gv
    pr = pr - dt * (lpv / 2 / jnp.sqrt(d))
    return q, p, r, pr, -v, -lpv


@functools.partial(jax.jit, static_argnames=("vfun", "d", "hbar", "m", "debug"))
def _drift_pois(q, p, r, pr, vfun, dt, d, hbar, m, debug):
    q = q + dt / m * p
    a, b = rpr2ab(r, pr, d, hbar if hbar > 0 else 1)
    # if debug:
    #     jax.debug.print("a,b: {a}, {b}", a=a, b=b)
    c = a + 1j * b
    c = c / (1.0 + dt * c / m)
    a = jnp.real(c)
    b = jnp.imag(c)
    # if debug:
    #     jax.debug.print("a,b: {a}, {b}", a=a, b=b)
    r, pr = ab2rpr(a, b, d, hbar if hbar > 0 else 1)
    return q, p, r, pr


@functools.partial(jax.jit, static_argnames=("log_like", "d", "hbar", "m", "n", "debug"))
def _leapfrog(log_like, x, p, r, pr, m, hbar, h, d, n, debug):
    # we use the usual way the KDK loops
    # are unrolled
    # for four iterations we have
    #
    #   K2 D K2, K2 D K2, K2 D K2, K2 D K2
    #   K2 D K      D K      D K      D K2
    #
    # so we get a single half kick, 4-1 = 3 full
    # drifts+kicks, then a full drift + halkf kick

    def vfun(x):
        return -1.0 * log_like(x)

    if debug:
        jax.debug.print("x,p,r,pr: {x} {p} {r} {pr}", x=x, p=p, r=r, pr=pr)

    # first half kick
    x, p, r, pr, nvi, nlpvi = _kick_pois(x, p, r, pr, vfun, h / 2.0, d, hbar, m)
    # if debug:
    #     jax.debug.print("x,p,r,pr: {x} {p} {r} {pr}", x=x, p=p, r=r, pr=pr)

    # n - 1 full drft + kick
    # for _ in range(n - 1):
    #     x, p, r, pr = _drift(log_like, x, p, r, pr, m, hbar, h, sqrtd, eps)
    #     x, p, r, pr, _, _ = _kick(log_like, x, p, r, pr, m, hbar, h, sqrtd, eps)

    def _dk(dummy, iv):
        x, p, r, pr = iv
        x, p, r, pr = _drift_pois(x, p, r, pr, vfun, h, d, hbar, m, debug)
        # if debug:
        #     jax.debug.print("x,p,r,pr: {x} {p} {r} {pr}", x=x, p=p, r=r, pr=pr)
        x, p, r, pr, _, _ = _kick_pois(x, p, r, pr, vfun, h, d, hbar, m)
        # if debug:
        #     jax.debug.print("x,p,r,pr: {x} {p} {r} {pr}", x=x, p=p, r=r, pr=pr)
        return x, p, r, pr

    x, p, r, pr = jax.lax.fori_loop(0, n-1, _dk, (x, p, r, pr))

    # full drift, half kick
    x, p, r, pr = _drift_pois(x, p, r, pr, vfun, h / 2.0, d, hbar, m, debug)
    # if debug:
    #     jax.debug.print("x,p,r,pr: {x} {p} {r} {pr}", x=x, p=p, r=r, pr=pr)
    x, p, r, pr, nvf, nlpvf = _kick_pois(x, p, r, pr, vfun, h / 2.0, d, hbar, m)
    if debug:
        jax.debug.print("x,p,r,pr: {x} {p} {r} {pr}", x=x, p=p, r=r, pr=pr)

    # reverse p, pr
    p = -p
    pr = -pr

    has_nans = (
        jnp.any(jnp.isnan(x))
        | jnp.any(jnp.isnan(p))
        | jnp.any(jnp.isnan(r))
        | jnp.any(jnp.isnan(pr))
        | jnp.any(jnp.isnan(nvi))
        | jnp.any(jnp.isnan(nvf))
    )
    if debug:
        jax.debug.print("has_nans: {has_nans}", has_nans=has_nans)
    nvf = jax.lax.cond(
        has_nans,
        lambda _x: -jnp.inf,
        lambda _x: _x,
        nvf,
    )
    nlpvf = jax.lax.cond(
        has_nans,
        lambda _x: -jnp.inf,
        lambda _x: _x,
        nlpvf,
    )
    p = jax.lax.cond(
        has_nans,
        lambda _x: jnp.zeros_like(_x),
        lambda _x: _x,
        p,
    )
    pr = jax.lax.cond(
        has_nans,
        lambda _x: jnp.zeros_like(_x),
        lambda _x: _x,
        pr,
    )

    return x, p, r, pr, nvi, nvf, nlpvi, nlpvf


def _hamiltonian_classical(neg_pot, p, m):
    return -neg_pot + p*p / 2.0 / m


def _delta_hamiltonian_quantum(neg_lap_pot, r, pr, hbar, m, sqrtd, d):
    return r / 2.0 / sqrtd * (
        -neg_lap_pot + 4.0 * pr * pr / m + hbar * hbar * d / 4 / m / r / r
    )


@functools.partial(jax.jit, static_argnames=("log_like", "h", "d", "hbar", "m", "n", "debug"))
def _step(log_like, m, hbar, h, d, n, debug, carry, not_used):
    (
        params,
        rng_key,
    ) = carry

    x, r, pri, logw = params
    sqrtd = np.sqrt(d)

    rng_key, nrm_key = jrng.split(rng_key)
    h = (jrng.uniform(nrm_key) * 0.1 + 1.0) * h

    # draw p, pr
    rng_key, nrm_key = jrng.split(rng_key)
    p = jrng.normal(nrm_key, shape=(d,)) * jnp.sqrt(m)
    rng_key, nrm_key = jrng.split(rng_key)
    prf = jrng.normal(nrm_key, shape=(1,)) * jnp.sqrt(m * sqrtd / 4 / r)
    xf, pf, rf, prf, nvi, nvf, nlpvi, nlpvf = _leapfrog(log_like, x, p, r, prf, m, hbar, h, d, n, debug)

    # compute acceptance prob and weight
    hci = _hamiltonian_classical(
        nvi, p, m
    )
    if hbar > 0:
        dhqi = _delta_hamiltonian_quantum(
            nlpvi, r, pri, hbar, m, sqrtd, d,
        )
        dhqi = jax.lax.cond(
            jnp.isinf(nlpvi),
            lambda _x: jnp.array([jnp.inf]),
            lambda _x: _x,
            dhqi,
        )
    else:
        dhqi = 0.0
    hqi = hci + dhqi
    hcf = _hamiltonian_classical(
        nvf, pf, m
    )
    if hbar > 0:
        dhqf = _delta_hamiltonian_quantum(
            nlpvf, rf, prf, hbar, m, sqrtd, d,
        )
        dhqf = jax.lax.cond(
            jnp.isinf(nlpvf),
            lambda _x: jnp.array([jnp.inf]),
            lambda _x: _x,
            dhqf,
        )
    else:
        dhqf = 0.0
    logwf = dhqf
    hqf = hcf + dhqf

    if debug:
        jax.debug.print("hqi: {hqi}", hqi=hqi)
        jax.debug.print("hqf: {hqf}", hqf=hqf)
    logq = hqi - hqf
    logq = jnp.clip(logq, min=-jnp.inf, max=0.0)
    q = jnp.exp(logq)

    # draw r from u[0,1]
    rng_key, unf_key = jrng.split(rng_key)
    racc = jrng.uniform(unf_key)

    # accept xn if r <= q else accept x
    acc_bool = racc <= q
    x_new = jnp.where(acc_bool, xf, x)
    r_new = jnp.where(acc_bool, rf, r)
    pr_new = jnp.where(acc_bool, prf, pri)
    logw_new = jnp.where(acc_bool, logwf, logw)
    acc_new = q
    ll_new = jnp.where(acc_bool, nvf, nvi)

    return (
        (x_new, r_new, pr_new, logw_new),
        rng_key,
    ), (x_new, logw_new, acc_new, ll_new, r_new)


def qtmc(
    rng_key,
    log_likelihood,
    n_dims,
    n_samples,
    m=1.0,
    hbar=1.0,
    params_init=None,
    leapfrog_step_size=None,
    n_leapfrog_steps=None,
    verbose=True,
    debug=False,
):
    """Run quantum tunelling HMC.

    Parameters
    ----------
    rng_key : PRNG key
        The RNG key to use.
    log_likelihood :  callable
        A callable with signature `log_likelihood(x)`
        that returns the log-likelihood for a single sample point.
    n_dims : int
        The number of dimensions.
    n_samples : int
        The number of sampling steps to take.
    params_init : jax.numpy.ndarray, optional
        The initial starting points for the walkers w/ shape (n_walkers, n_dims).
        If `None`, then a small unit normal ball about zero is made.
    leapfrog_step_size : float, optional
        The step size for the leapfrog integration. The default of `None` will
        use `0.1 * (n_dims)**(-0.25)`.
    n_leapfrog_steps : int, optional
        The number of leapfrog steps to take during the inegration. If `None`, a
        default of `int(1/leapfrog_step_size)` will be used.
    verbose : bool, optional
        If True, print the progress of the chain and the acceptance rate. The default
        is True.

    Returns
    -------
    chain : jax.numpy.ndarray
        The MCMC chain w/ shape (n_samples, n_dims).
    log_weights : jax.numpy.ndarray
        The logarithm of the weights for each sample in the chain w/ shape (n_samples,).
    acc : jax.numpy.ndarray
        The acceptance probabilities w/ shape (n_samples,).
    loglike : jax.numpy.ndarray
        The log-likelihood at each step w/ shape (n_samples,).
    """
    if leapfrog_step_size is None:
        leapfrog_step_size = 0.1 * np.power(n_dims, -0.25)

    if n_leapfrog_steps is None:
        n_leapfrog_steps = int(1 / leapfrog_step_size)

    if params_init is None:
        rng_key, init_key = jrng.split(rng_key)
        params_init = jrng.normal(init_key, shape=(n_dims,)) / jnp.sqrt(m)
    else:
        assert n_dims == params_init.shape[0], (
            "The number of dimensions given by `n_dims` must match the number of "
            "dimensions implied by `params_init`. You passed `n_dims`={n_dims} and "
            f"`params_init.shape[0]={params_init.shape[0]}."
        )
    rng_key, init_key = jrng.split(rng_key)
    r_init = jnp.array([1 / m]) * (jrng.uniform(init_key) * 0.1 + 1.0)
    rng_key, nrm_key = jrng.split(rng_key)
    pr_init = jrng.normal(nrm_key, shape=(1,)) * jnp.sqrt(m * jnp.sqrt(n_dims) / 4 / r_init)
    if hbar > 0:
        lval_init, _, lplval_init, _ = _value_and_grad_forward_laplacian(params_init, log_likelihood)

        qpot = lplval_init
        logw_init = _delta_hamiltonian_quantum(
            qpot, r_init, pr_init, hbar, m, jnp.sqrt(n_dims), n_dims,
        )
    else:
        logw_init = jnp.array([0.0])

    _local_step = _step

    _local_step = functools.partial(
        _local_step,
        log_likelihood,
        m,
        hbar,
        leapfrog_step_size,
        n_dims,
        n_leapfrog_steps,
        debug,
    )

    # if not debug:
    if verbose:
        _local_step = scan_tqdm(
            n_samples+1, tqdm_type="std", ncols=80, desc="sampling"
        )(_local_step)

    _, (chain, logwgts, acc, loglike, r) = jax.lax.scan(
        _local_step, ((params_init, r_init, pr_init, logw_init), rng_key), xs=jnp.arange(n_samples+1)
    )
    # else:
    #     import tqdm
    #     ret = []
    #     state = ((params_init, r_init, pr_init, logw_init), rng_key)
    #     for x in tqdm.trange(n_samples+1):
    #         state, _ret = _local_step(state, x)
    #         ret.append(_ret)

    #     ret = jnp.swapaxes(jnp.array(ret), 0, 1)
    #     chain, logwgts, acc, loglike, r = ret

    if verbose:
        print("acceptance rate: %0.2f%%" % (100.0 * acc[1:, ].mean()))
    return chain[1:, ], logwgts[1:, ], acc[1:, ], loglike[1:, ], r[1:, ]
