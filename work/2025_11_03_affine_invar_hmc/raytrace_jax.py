###**************************************
### Ray Tracing Sampler
### Original HMC Implementation: Copyright (C) 2024, Martin Marek
### See original source at https://github.com/martin-marek/mini-hmc-jax
### Additional Changes (sample_raytracer, UpdateV, raytracer_leapfrog): Copyright (C) 2025, Peter Behroozi
###
### Licensed under the Apache License, Version 2.0 (the "License");
### you may not use this file except in compliance with the License.
### You may obtain a copy of the License at
###
###     http://www.apache.org/licenses/LICENSE-2.0
###
### Unless required by applicable law or agreed to in writing, software
### distributed under the License is distributed on an "AS IS" BASIS,
### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
### See the License for the specific language governing permissions and
### limitations under the License.
###***************************************/


import os
os.environ['ENABLE_PJRT_COMPATIBILITY']='1'

import operator as op
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_leaves, tree_reduce

__all__ = ["sample_raytrace", "sample_hamiltonian"]

def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None: treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def normal_like_tree(rng_key, target, mean=0, std=1):
    # https://github.com/google/jax/discussions/9508#discussioncomment-2144076
    keys_tree = random_split_like_tree(rng_key, target)
    return tree_map(lambda l, k: mean + std*jax.random.normal(k, l.shape, l.dtype), target, keys_tree)


def ifelse(cond, val_true, val_false):
    return jax.lax.cond(cond, lambda x: x[0], lambda x: x[1], (val_true, val_false))


def ScatterV(momentum, refresh_rate, dt, key):
    key, normal_key = jax.random.split(key, 2)
    f=jnp.exp(-jnp.abs(refresh_rate*dt))
    fn=jnp.sqrt(1.0-f*f)
    diffusion = normal_like_tree(normal_key, momentum)
    momentum = tree_map(lambda m, d: f*m + fn*d, momentum, diffusion)
    return momentum, key


def hmc_leapfrog_refresh(params, momentum, log_prob_fn, step_size, n_steps, refresh_rate, key):
    """Approximates Hamiltonian dynamics using the leapfrog algorithm."""

    kinetic_energy_diff = 0
    
    # define a single step
    def step(i, args):
        params, momentum, kinetic_energy_diff, key = args

        # scatter velocity
        momentum, key = ScatterV(momentum, refresh_rate, step_size/2.0, key)

        # update params
        params = tree_map(lambda p, m: p + 0.5 * m * step_size, params, momentum)

        # update momentum
        momentum_dot = tree_reduce(op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(momentum)))
        grad = jax.grad(log_prob_fn)(params)
        momentum = tree_map(lambda m, g: m + step_size * g, momentum, grad)
        new_momentum_dot = tree_reduce(op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(momentum)))
        kinetic_energy_diff += 0.5*(momentum_dot - new_momentum_dot)

        # update params
        params = tree_map(lambda p, m: p + 0.5 * m * step_size, params, momentum)

        # scatter velocity
        momentum, key = ScatterV(momentum, refresh_rate, step_size/2.0, key)
        return params, momentum, kinetic_energy_diff, key 

    # do 'n_steps'
    new_params, new_momentum, kinetic_energy_diff, key = jax.lax.fori_loop(0, n_steps, step, (params, momentum, kinetic_energy_diff, key))

    return new_params, new_momentum, -kinetic_energy_diff, key


def hmc_leapfrog_norefresh(params, momentum, log_prob_fn, step_size, n_steps, refresh_rate, key):
    """Approximates Hamiltonian dynamics using the leapfrog algorithm."""

    momentum_dot = tree_reduce(op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(momentum)))
    # define a single step
    def step(i, args):
        params, momentum = args

        # update params
        params = tree_map(lambda p, m: p + 0.5 * m * step_size, params, momentum)

        # update momentum
        grad = jax.grad(log_prob_fn)(params)
        momentum = tree_map(lambda m, g: m + step_size * g, momentum, grad)

        # update params
        params = tree_map(lambda p, m: p + 0.5 * m * step_size, params, momentum)
        return params, momentum

    # do 'n_steps'
    new_params, new_momentum = jax.lax.fori_loop(0, n_steps, step, (params, momentum))

    new_momentum_dot = tree_reduce(op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(new_momentum)))
    kinetic_energy_diff = 0.5*(momentum_dot - new_momentum_dot)
    return new_params, new_momentum, -kinetic_energy_diff, key



def sample_hamiltonian(key, params_init, log_prob_fn, n_steps, n_leapfrog_steps, step_size, refresh_rate=0, metro_check=1, sample_hmc=True):
    return sample_raytrace(key, params_init, log_prob_fn, n_steps, n_leapfrog_steps, step_size, refresh_rate, metro_check, sample_hmc)
    """
    Runs HMC and returns the full Markov chain as a Pytree.
    - params: array
    - log_prob_fn: function that takes params as the only argument and returns a scalar value
    """


def UpdateV(momentum, grad, D, step_size):
    norm_v = jnp.sqrt(tree_reduce(op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(momentum))))
    unit_v = tree_map(lambda m, n: m/n, momentum, norm_v)
    norm_g = jnp.sqrt(tree_reduce(op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(grad))))
    unit_g = tree_map(lambda g, n: g/n, grad, norm_g)
    sub_vec = tree_map(lambda v, g: v-g, unit_v, unit_g)
    sub_vec_norm = jnp.sqrt(tree_reduce(op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(sub_vec))))
    add_vec = tree_map(lambda v, g: v+g, unit_v, unit_g)
    add_vec_norm = jnp.sqrt(tree_reduce(op.add, tree_map(lambda x: (x**2).sum(), tree_leaves(add_vec))))
    #cases where norm_g==0 or norm_v==0 are already dealt with correctly with the numpy atan2
    theta_i = 2.0*jnp.arctan2(sub_vec_norm, add_vec_norm)
    theta_f = 2.0*jnp.arctan2(sub_vec_norm, add_vec_norm*jnp.exp(norm_g*norm_v*step_size/(D-1.0)))
    f_v = jax.lax.cond(jnp.sin(theta_i)==0,
                       lambda _: 1.0,
                       lambda _: jnp.sin(theta_f)/jnp.sin(theta_i),
                       operand=None)
    f_n = (jnp.cos(theta_f) - f_v*jnp.cos(theta_i))*norm_v
    new_momentum = tree_map(lambda m, ug, fv, fn: fv*m+fn*ug, momentum, unit_g, f_v, f_n)
    delta_ln_L = jax.lax.cond(jnp.sin(theta_i)==0,
                              lambda _: norm_v*norm_g*step_size*jnp.cos(theta_i),
                              lambda _: (1.0-D)*jnp.log(f_v),
                              operand=None)
    return (new_momentum, delta_ln_L, norm_v, norm_g, theta_i, theta_f, f_v, f_n)

def raytracer_leapfrog_refresh(params, momentum, log_prob_fn, step_size, n_steps, refresh_rate, key):
    """Approximates Ray tracing dynamics using the leapfrog algorithm."""
    ln_L = 0

    # define a single step
    def step(i, args):
        params, momentum, ln_L, key = args

        momentum, key = ScatterV(momentum, refresh_rate, step_size/2.0, key)
        
        # update params
        params = tree_map(lambda p, m: p + 0.5 * m * step_size, params, momentum)

        # update momentum
        grad = jax.grad(log_prob_fn)(params)
        momentum, delta_ln_L, norm_v, norm_g, theta_i, theta_f, f_v, f_n = UpdateV(momentum, grad, momentum.size, step_size)
        ln_L += delta_ln_L

        # update params
        params = tree_map(lambda p, m: p + 0.5 * m * step_size, params, momentum)

        momentum, key = ScatterV(momentum, refresh_rate, step_size/2.0, key)

        return params, momentum, ln_L, key

    # do 'n_steps'
    new_params, new_momentum, new_ln_L, key = jax.lax.fori_loop(0, n_steps, step, (params, momentum, ln_L, key))

    return new_params, new_momentum, new_ln_L, key

def raytracer_leapfrog_norefresh(params, momentum, log_prob_fn, step_size, n_steps, refresh_rate, key):
    """Approximates Ray tracing dynamics using the leapfrog algorithm."""
    ln_L = 0

    # define a single step
    def step(i, args):
        params, momentum, ln_L = args

        # update params
        params = tree_map(lambda p, m: p + 0.5 * m * step_size, params, momentum)

        # update momentum
        grad = jax.grad(log_prob_fn)(params)
        momentum, delta_ln_L, norm_v, norm_g, theta_i, theta_f, f_v, f_n = UpdateV(momentum, grad, momentum.size, step_size)
        ln_L += delta_ln_L

        # update params
        params = tree_map(lambda p, m: p + 0.5 * m * step_size, params, momentum)
        return params, momentum, ln_L

    # do 'n_steps'
    new_params, new_momentum, new_ln_L = jax.lax.fori_loop(0, n_steps, step, (params, momentum, ln_L))

    return new_params, new_momentum, new_ln_L, key


def sample_raytrace(key, params_init, log_prob_fn, n_steps, n_leapfrog_steps, step_size, refresh_rate=0, sample_hmc = False, metro_check=1):
    """
    Runs Ray Tracing and returns the full Markov chain as a Pytree.
    - params: array
    - log_prob_fn: function that takes params as the only argument and returns a scalar value
    """
    
    if (params_init.size < 2 and sample_hmc==False):
        print('[Warning] Ray tracing requires 2 or more dimensions.  Defaulting to use HMC instead.')
        sample_hmc=True
    leapfrog_func = raytracer_leapfrog_norefresh
    if (sample_hmc):
        if (refresh_rate):
            leapfrog_func = hmc_leapfrog_refresh
        else:
            leapfrog_func = hmc_leapfrog_norefresh
    elif (refresh_rate):
        leapfrog_func = raytracer_leapfrog_refresh
    
    # define a single step
    def ray_step_fn(carry, x):
        params, key = carry
        key, normal_key, uniform_key = jax.random.split(key, 3)

        # generate random momentum
        momentum = normal_like_tree(normal_key, params)
        delta_ln_L = 0

        # leapfrog
        new_params, new_momentum, delta_ln_L, key = \
            leapfrog_func(params, momentum, log_prob_fn, step_size, n_leapfrog_steps, refresh_rate, key)

        # MH correction
        old_lnl = log_prob_fn(params)
        new_lnl = log_prob_fn(new_params) 
        log_likelihood_diff = new_lnl - old_lnl
        log_accept_prob = log_likelihood_diff - delta_ln_L
        log_accept_prob = jnp.nan_to_num(log_accept_prob, nan=-jnp.inf)
        accept_prob = jnp.minimum(1.0, jnp.exp(log_accept_prob))
        accept_prob = jnp.maximum(accept_prob, 1.0-metro_check)
        accept = jax.random.uniform(uniform_key) < accept_prob
        params = ifelse(accept, new_params, params)
        lnl = ifelse(accept, new_lnl, old_lnl)
        return (params, key), (params, accept_prob, lnl)

    # do 'n_steps'
    _, (chain, accept_prob, log_likelihood) = jax.lax.scan(ray_step_fn, (params_init, key), xs=None, length=n_steps)

    print(f'accept={accept_prob.mean():.2%}')
    return chain, log_likelihood
