import functools

import jax
import jax.numpy as jnp

from folx import forward_laplacian


@functools.partial(jax.jit, static_argnames="fun")
def value_and_grad_forward_laplacian(x, fun):
    def _fun_for_grad_flp(x):
        res = forward_laplacian(fun)(x)
        return res.laplacian, res
    grad_lp, res = jax.jacfwd(_fun_for_grad_flp, has_aux=True)(x)
    return res.x, res.jacobian.dense_array[0], res.laplacian, grad_lp


@functools.partial(jax.jit, static_argnames=("vfun", "dt", "d", "hbar", "m"))
def kick_c(q, p, b, pr, vfun, dt, d, hbar, m):
    v, gv, lpv, glpv = value_and_grad_forward_laplacian(q, vfun)
    p = p - dt * gv
    return q, p, b, pr


@functools.partial(jax.jit, static_argnames=("vfun", "dt", "d", "hbar", "m"))
def drift_c(q, p, b, pr, vfun, dt, d, hbar, m):
    q = q + dt / m * p
    return q, p, b, pr


@functools.partial(jax.jit, static_argnames=("vfun", "dt", "d", "hbar", "m"))
def evolve_state_c(q, p, r, pr, vfun, dt, d, hbar, m):
    q, p, r, pr = kick_c(q, p, r, pr, vfun, dt/2.0, d, hbar, m)
    q, p, r, pr = drift_c(q, p, r, pr, vfun, dt, d, hbar, m)
    q, p, r, pr = kick_c(q, p, r, pr, vfun, dt/2.0, d, hbar, m)
    return q, p, r, pr


@functools.partial(jax.jit, static_argnames=("vfun", "dt", "d", "hbar", "m"))
def kick_symp(q, p, r, pr, vfun, dt, d, hbar, m):
    v, gv, lpv, glpv = value_and_grad_forward_laplacian(q, vfun)
    p = p - dt * (gv + r / 2 / jnp.sqrt(d) * glpv)
    pr = pr - dt * (lpv / 2 / jnp.sqrt(d) - jnp.power(d, 1.5) * hbar * hbar / 8 / m / r / r)
    return q, p, r, pr


@functools.partial(jax.jit, static_argnames=("vfun", "dt", "d", "hbar", "m"))
def drift_symp(q, p, r, pr, vfun, dt, d, hbar, m):
    q = q + dt / m * p
    pr_fac = 1.0 + dt * 2.0 / m / jnp.sqrt(d) * pr
    r = r * pr_fac * pr_fac
    pr = pr / pr_fac

    return q, p, r, pr


@functools.partial(jax.jit, static_argnames=("vfun", "dt", "d", "hbar", "m"))
def integ_symp(q, p, r, pr, vfun, dt, d, hbar, m):
    q, p, r, pr = kick_symp(q, p, r, pr, vfun, dt/2.0, d, hbar, m)
    q, p, r, pr = drift_symp(q, p, r, pr, vfun, dt, d, hbar, m)
    q, p, r, pr = kick_symp(q, p, r, pr, vfun, dt/2.0, d, hbar, m)
    return q, p, r, pr


def rpr2ab(r, pr, d, hbar):
    a = 2.0 / jnp.sqrt(d) * pr
    b = hbar * jnp.sqrt(d) / 2 / r
    return a, b


def ab2rpr(a, b, d, hbar):
    pr = jnp.sqrt(d) / 2.0 * a
    r = hbar * jnp.sqrt(d) / 2 / b
    return r, pr


@functools.partial(jax.jit, static_argnames=("vfun", "dt", "d", "hbar", "m"))
def kick_pois(q, p, r, pr, vfun, dt, d, hbar, m):
    v, gv, lpv, glpv = value_and_grad_forward_laplacian(q, vfun)
    p = p - dt * (gv + r / 2 / jnp.sqrt(d) * glpv)
    pr = pr - dt * (lpv / 2 / jnp.sqrt(d))
    return q, p, r, pr


@functools.partial(jax.jit, static_argnames=("vfun", "dt", "d", "hbar", "m"))
def drift_pois(q, p, r, pr, vfun, dt, d, hbar, m):
    q = q + dt / m * p
    a, b = rpr2ab(r, pr, d, hbar)
    c = a + 1j * b
    c = c / (1.0 + dt * c / m)
    a = jnp.real(c)
    b = jnp.imag(c)
    r, pr = ab2rpr(a, b, d, hbar)
    return q, p, r, pr


@functools.partial(jax.jit, static_argnames=("vfun", "dt", "d", "hbar", "m"))
def evolve_state_pois(q, p, r, pr, vfun, dt, d, hbar, m):
    q, p, r, pr = kick_pois(q, p, r, pr, vfun, dt/2.0, d, hbar, m)
    q, p, r, pr = drift_pois(q, p, r, pr, vfun, dt, d, hbar, m)
    q, p, r, pr = kick_pois(q, p, r, pr, vfun, dt/2.0, d, hbar, m)
    return q, p, r, pr


@functools.partial(jax.jit, static_argnames=("vfun", "dt", "d", "hbar", "m", "eps"))
def kick_pois_ab(q, p, a, b, vfun, dt, d, hbar, m, eps=0.0):
    v, gv, lpv, glpv = value_and_grad_forward_laplacian(q, vfun)
    bpe = b - eps
    p = p - dt * (gv + hbar / 4 / bpe * glpv)
    a = a - dt * (lpv / d)
    return q, p, a, b


@functools.partial(jax.jit, static_argnames=("vfun", "dt", "d", "hbar", "m", "eps"))
def drift_pois_ab(q, p, a, b, vfun, dt, d, hbar, m, eps=0.0):
    bpe = b - eps
    q = q + dt / m * p
    c = a + 1j * bpe
    c = c / (1.0 + dt * c / m)
    a = jnp.real(c)
    bpe = jnp.imag(c)
    return q, p, a, bpe + eps


@functools.partial(jax.jit, static_argnames=("vfun", "dt", "d", "hbar", "m", "eps"))
def evolve_state_pois_ab(q, p, a, b, vfun, dt, d, hbar, m, eps=0.0):
    q, p, a, b = kick_pois_ab(q, p, a, b, vfun, dt/2.0, d, hbar, m, eps=eps)
    q, p, a, b = drift_pois_ab(q, p, a, b, vfun, dt, d, hbar, m, eps=eps)
    q, p, a, b = kick_pois_ab(q, p, a, b, vfun, dt/2.0, d, hbar, m, eps=eps)
    return q, p, a, b
