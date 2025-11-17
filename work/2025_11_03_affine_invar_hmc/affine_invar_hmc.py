import functools

import jax
import jax.numpy as jnp
import jax.random as jrng
import numpy as np

from jax_tqdm import scan_tqdm


def _leapfrog_base(neg_log_like, x, p, B, n, h, scalar):
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

    # first half kick
    vi, g = vgfunc(x)
    p = p - h / 2.0 * jnp.dot(B.T, g)

    # n - 1 full drft + kick
    for _ in range(n - 1):
        if scalar:
            x = x + h * B * p
        else:
            x = x + h * jnp.dot(B, p)
        g = gfunc(x)
        p = p - h * jnp.dot(B.T, g)

    # full drift, half kick
    if scalar:
        x = x + h * B * p
    else:
        x = x + h * jnp.dot(B, p)
    vf, g = vgfunc(x)
    p = p - h / 2 * jnp.dot(B.T, g)

    # reverse p
    p = -p

    has_nans = jnp.any(jnp.isnan(p)) | jnp.any(jnp.isnan(p)) | jnp.any(jnp.isnan(vi)) | jnp.any(jnp.isnan(vf))
    vf = jax.lax.cond(
        has_nans,
        lambda _x: jnp.inf,
        lambda _x: _x,
        vf,
    )
    p = jax.lax.cond(
        has_nans,
        lambda _x: jnp.zeros_like(_x),
        lambda _x: _x,
        p,
    )

    return x, p, vi, vf


@functools.partial(
    jax.jit,
    static_argnums=(0, 1, 2, 3, 4),
)
def _walk_step(
    neg_log_like,
    n_steps, 
    h, 
    n_walkers_2,
    n_dims,
    carry,
    x_not_used,
):
    (
        params,
        rng_key,
    ) = carry

    _leapfrog = jax.jit(
        jax.vmap(
            _leapfrog_base,
            in_axes=(None, 0, 0, None, None, None, None),
            out_axes=0,
        ),
        static_argnums=(0, 4, 6),
    )

    bfac = 1.0 / np.sqrt(n_walkers_2)

    x_new = []
    acc_new = []
    nll_new = []

    for s in range(2):
        # draw p
        rng_key, nrm_key = jrng.split(rng_key)
        p = jrng.normal(nrm_key, shape=(n_walkers_2, n_walkers_2))
        
        if s == 0:
            x = params[ :n_walkers_2, :]
            xB = params[n_walkers_2:, :]
        else:
            x = params[n_walkers_2:, :]
            # use chain from previous s loop
            xB = x_new[0]

        # make B
        mn = jnp.mean(xB, axis=0, keepdims=True)
        B = (xB - mn) * bfac
        B = B.T

        # do _leapfrog
        x_pr, p_pr, v, v_pr = _leapfrog(neg_log_like, x, p, B, n_steps, h, False)

        # measure q = exp(-V(xn) - 0.5 * pn * pn + V(x) + 0.5 * p *p)
        logq = v + 0.5 * jnp.sum(p * p, axis=1) - v_pr - 0.5 * jnp.sum(p_pr * p_pr, axis=1)
        q = jnp.exp(logq)
        q = jnp.clip(q, min=0, max=1)
        
        # draw r from u[0,1]
        rng_key, unf_key = jrng.split(rng_key)
        r = jrng.uniform(unf_key, shape=(n_walkers_2,))

        # accept xn if r <= q else accept x
        acc_val = r <= q
        x_new.append(jnp.where(acc_val.reshape(n_walkers_2, 1), x_pr, x))
        acc_new.append(q)
        nll_new.append(jnp.where(acc_val, v_pr, v))

    x_new = jnp.concatenate(x_new)
    acc_new = jnp.concatenate(acc_new)
    nll_new = jnp.concatenate(nll_new)

    return (
        x_new,
        rng_key, 
    ), (x_new, acc_new, nll_new)



def ensemble_hmc(neg_log_like, x_init, n_samples, n_dims, n_walkers, h, n_steps, rng_key):
    # V = -log_like(x)
    # we carry the step, ensemble / batch dimensions at the start so that
    # the returned chain is shaped [steps, n_walkers, n_dims]
    assert n_walkers % 2 == 0
    n_walkers_2 = n_walkers // 2
    
    _local_walk_step = scan_tqdm(n_samples,  tqdm_type='std', ncols=80, desc="sampling")(functools.partial(
        _walk_step,
        neg_log_like, 
        n_steps, 
        h, 
        n_walkers_2,
        n_dims,
    ))
    
    _, (chain, acc, nloglike) = jax.lax.scan(_local_walk_step, (x_init, rng_key), xs=jnp.arange(n_samples))    
    print("acceptance rate: %0.2f%%" % (100.0 * acc.mean()))
    return chain, acc, nloglike
