import functools

import jax
import jax.numpy as jnp
import jax.random as jrng

from jax_tqdm import scan_tqdm


@functools.partial(
    jax.jit,
    static_argnums=(0, 3, 4, 5),
)
def _constr_leapfrog_base(neg_log_like, x, p, n, h, constraint):
    # we use the usual way the KDK loops
    # are unrolled
    # for four iterations we have
    #
    #   K2 D K2, K2 D K2, K2 D K2, K2 D K2
    #   K2 D K      D K      D K      D K2
    #
    # so we get a single half kick, 4-1 = 3 full
    # drifts+kicks, then a full drift + halkf kick

    gfunc = jax.grad(neg_log_like)
    vgfunc = jax.value_and_grad(neg_log_like)
    gcfunc = jax.grad(constraint)

    def _reflect(_x, _p):
        n = gcfunc(_x)
        n = n / jnp.sqrt(jnp.sum(n * n))
        _p = _p - 2.0 * jnp.dot(_p, n) * n
        return _x, _p

    def _kick(_x, _p):
        g = gfunc(_x)
        _p = _p - h * g
        return _x, _p

    # first half kick
    vi, g = vgfunc(x)
    p = p - h / 2.0 * g

    # n - 1 full drft + kick
    for _ in range(n - 1):
        x = x + h * p

        x, p = jax.lax.cond(
            constraint(x) >= 0,
            lambda x, p: _kick(x, p),
            lambda x, p: _reflect(x, p),
            x,
            p,
        )

    # full drift, half kick
    x = x + h * p
    vf, g = vgfunc(x)
    p = p - h / 2 * g

    # reverse p
    p = -p

    # if the state has nans or violates the constraint,
    # return its neg log-like as inf and set p/x to zero
    has_nans_or_bad_constr = (
        jnp.any(jnp.isnan(p)) 
        | jnp.any(jnp.isnan(p)) 
        | jnp.any(jnp.isnan(vi)) 
        | jnp.any(jnp.isnan(vf))
        | jnp.any(jnp.isnan(x))
        | jnp.any(constraint(x) < 0)
    )
    vf, p, x = jax.lax.cond(
        has_nans_or_bad_constr,
        lambda _vf, _p, _x: (jnp.inf, jnp.zeros_like(_p), jnp.zeros_like(_x)),
        lambda _vf, _p, _x: (_vf, _p, _x),
        vf,
        p,
        x
    )

    return x, p, vi, vf


@functools.partial(
    jax.jit,
    static_argnums=(0, 1, 2, 3, 4),
)
def _constr_step(
    neg_log_like,
    n_steps, 
    h, 
    n_dims,
    constraint,
    carry,
    x_not_used,
):
    (
        params,
        rng_key,
    ) = carry

    # draw p
    rng_key, nrm_key = jrng.split(rng_key)
    p = jrng.normal(nrm_key, shape=(n_dims,))
        
    # do _leapfrog
    x_pr, p_pr, v, v_pr = _constr_leapfrog_base(neg_log_like, params, p, n_steps, h, constraint)
        
    # measure q = exp(-V(xn) - 0.5 * pn * pn + V(x) + 0.5 * p *p)
    logq = v + 0.5 * jnp.sum(p * p) - v_pr - 0.5 * jnp.sum(p_pr * p_pr)
    q = jnp.exp(logq)
    q = jnp.clip(q, min=0, max=1)
        
    # draw r from u[0,1]
    rng_key, unf_key = jrng.split(rng_key)
    r = jrng.uniform(unf_key)

    # accept xn if r <= q else accept x
    x_new, nll_new = jax.lax.cond(
        r <= q,
        lambda _x, _y: _x,
        lambda _x, _y: _y,
        (x_pr, v_pr),
        (params, v),
    )
    acc_new = q

    return (
        x_new,
        rng_key, 
    ), (x_new, acc_new, nll_new)



def constr_hmc(neg_log_like, x_init, n_samples, n_dims, h, n_steps, constraint, rng_key):
    # V = -log_like(x)
    
    _local_constr_step = scan_tqdm(n_samples,  tqdm_type='std', ncols=80, desc="sampling")(functools.partial(
        _constr_step,
        neg_log_like, 
        n_steps, 
        h, 
        n_dims,
        constraint,
    ))
    
    _, (chain, acc, nloglike) = jax.lax.scan(_local_constr_step, (x_init, rng_key), xs=jnp.arange(n_samples))    
    print("acceptance rate: %0.2f%%" % (100.0 * acc.mean()))
    return (
        jnp.expand_dims(chain, axis=1), 
        acc, 
        nloglike,
    )
