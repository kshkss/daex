import gc
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from daex.semi_explicit import (
    SemiExplicitDAE,
    _const_fn,
    _deriv_fn,
    _jac_fn,
    _make_model,
    _ravel_pytree,
    _res_fn,
    def_semi_explicit_dae,
)

jax.config.update("jax_enable_x64", True)


class State(NamedTuple):
    x: jax.Array
    y: jax.Array


class Params(NamedTuple):
    a: jax.Array


@jax.jit
def derivative(params: Params, t: jax.Array, stat: State) -> State:
    return State(
        x=None,
        y=stat.x * params.a * t,
    )


@jax.jit
def constraint(params: Params, t: jax.Array, stat: State) -> jax.Array:
    return stat.x**2 - jnp.sum(stat.y)


def make_problem(shape=()):
    params = Params(a=jnp.full(shape, 2.0))
    t0 = jnp.asarray(0.5)
    xy0 = State(x=jnp.asarray(1.0), y=jnp.full(shape, 1.0))
    return params, t0, xy0


def call_model_fns(dae, params, t, xy):
    x0, y0 = dae.partition(xy)
    model, x, y, a = _make_model(dae, x0, y0, params)
    yp = _deriv_fn(model, t, x, y, a)
    xy = jnp.concatenate([x, y])
    xyp = jnp.concatenate([jnp.zeros_like(x), yp])
    return (
        model,
        _res_fn(model, t, xy, xyp, a),
        _jac_fn(model, t, xy, xyp, 1.0, a),
    )


def test_model_fns_match_user_fns():
    params, t0, xy0 = make_problem()
    dae = def_semi_explicit_dae(derivative, constraint, params, t0, xy0)
    x0, y0 = dae.partition(xy0)
    model, x, y, a = _make_model(dae, x0, y0, params)

    yp, _ = ravel_pytree(derivative(params, t0, xy0))
    g, _ = ravel_pytree(constraint(params, t0, xy0))
    np.testing.assert_allclose(_deriv_fn(model, t0, x, y, a), yp)
    np.testing.assert_allclose(_const_fn(model, t0, x, y, a), g)


def test_cache_is_reused_across_daes():
    params, t0, xy0 = make_problem()
    dae1 = def_semi_explicit_dae(derivative, constraint, params, t0, xy0)
    model1, _, _ = call_model_fns(dae1, params, t0, xy0)
    deriv_size = _res_fn._cache_size()
    const_size = _jac_fn._cache_size()

    params, t0, xy0 = make_problem()
    dae2 = def_semi_explicit_dae(derivative, constraint, params, t0, xy0)
    model2, _, _ = call_model_fns(dae2, params, t0, xy0)

    assert model1 == model2
    assert hash(model1) == hash(model2)
    assert _res_fn._cache_size() == deriv_size
    assert _jac_fn._cache_size() == const_size


def test_cache_entry_per_structure():
    params, t0, xy0 = make_problem()
    dae = def_semi_explicit_dae(derivative, constraint, params, t0, xy0)
    model1, _, _ = call_model_fns(dae, params, t0, xy0)
    deriv_size = _res_fn._cache_size()
    const_size = _jac_fn._cache_size()

    params, t0, xy0 = make_problem(shape=(2,))
    dae = def_semi_explicit_dae(derivative, constraint, params, t0, xy0)
    model2, _, _ = call_model_fns(dae, params, t0, xy0)

    assert model1 != model2
    assert _res_fn._cache_size() == deriv_size + 1
    assert _jac_fn._cache_size() == const_size + 1


def test_cache_survives_dae_disposal():
    params, t0, xy0 = make_problem()
    dae = def_semi_explicit_dae(derivative, constraint, params, t0, xy0)
    call_model_fns(dae, params, t0, xy0)
    deriv_size = _res_fn._cache_size()
    const_size = _jac_fn._cache_size()
    assert deriv_size > 0
    assert const_size > 0

    del dae
    gc.collect()
    assert _res_fn._cache_size() == deriv_size
    assert _jac_fn._cache_size() == const_size


def test_with_statement_keeps_cache():
    params, t0, xy0 = make_problem()
    with def_semi_explicit_dae(derivative, constraint, params, t0, xy0) as dae:
        assert isinstance(dae, SemiExplicitDAE)
        call_model_fns(dae, params, t0, xy0)
        deriv_size = _res_fn._cache_size()
        const_size = _jac_fn._cache_size()
    assert _res_fn._cache_size() == deriv_size
    assert _jac_fn._cache_size() == const_size


def test_unravel_of_empty_pytree_is_comparable():
    _, unravel1 = _ravel_pytree(State(x=None, y=None))
    _, unravel2 = _ravel_pytree(State(x=None, y=None))
    assert unravel1 == unravel2
    assert hash(unravel1) == hash(unravel2)
    assert unravel1(jnp.zeros(0)) == State(x=None, y=None)
