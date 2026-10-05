import dataclasses
import gc
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from sksundae._cy_ida import IDA

from daex.semi_explicit import (
    _jac_fn,
    _res_fn,
    daeint,
    def_semi_explicit_dae,
)

jax.config.update("jax_enable_x64", True)

LEGACY_CALLBACKS = [
    "deriv_fn",
    "const_fn",
    "resfn",
    "jacfn",
    "deriv_ext",
    "resfn_ext",
    "jacfn_ext",
    "deriv_adj",
    "const_adj",
    "resfn_adj",
    "jacfn_adj",
    "da_fn",
]

OPTIONS = {"rtol": 1e-8, "atol": 1e-10}


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


def make_problem():
    params = Params(a=jnp.asarray(2.0))
    t0 = jnp.asarray(0.5)
    xy0 = State(x=jnp.asarray(1.0), y=jnp.asarray(1.0))
    return params, t0, xy0


def test_legacy_attributes_exist():
    params, t0, xy0 = make_problem()
    dae = def_semi_explicit_dae(derivative, constraint, params, t0, xy0)
    assert dae.x_size == 1
    for name in LEGACY_CALLBACKS + ["_clear_cache"]:
        assert callable(getattr(dae, name)), name


def test_legacy_deriv_and_const_match_user_fns():
    params, t0, xy0 = make_problem()
    dae = def_semi_explicit_dae(derivative, constraint, params, t0, xy0)
    x0, y0 = dae.partition(xy0)
    a, _ = ravel_pytree(params)
    x, _ = ravel_pytree(x0)
    y, _ = ravel_pytree(y0)

    yp, _ = ravel_pytree(derivative(params, t0, xy0))
    g, _ = ravel_pytree(constraint(params, t0, xy0))
    np.testing.assert_allclose(dae.deriv_fn(a, t0, x, y), yp)
    np.testing.assert_allclose(dae.const_fn(a, t0, x, y), g)


def test_legacy_resfn_and_jacfn_solve_like_daeint():
    params, t0, xy0 = make_problem()
    dae = def_semi_explicit_dae(derivative, constraint, params, t0, xy0)
    ts = jnp.linspace(t0, 1.5, 5)
    x0, y0 = dae.partition(xy0)
    a, _ = ravel_pytree(params)
    x, _ = ravel_pytree(x0)
    y, _ = ravel_pytree(y0)
    yp = dae.deriv_fn(a, t0, x, y)

    ida = IDA(
        dae.resfn,
        jacfn=dae.jacfn,
        userdata=(a,),
        algebraic_idx=np.arange(dae.x_size),
        **OPTIONS,
    )
    results = ida.solve(
        np.asarray(ts),
        np.append(x, y),
        np.append(np.zeros_like(x), yp),
    )
    assert results.success

    expected = daeint(params, dae, ts, xy0, options=OPTIONS).values
    np.testing.assert_allclose(results.y[:, 0], expected.x, rtol=1e-6)
    np.testing.assert_allclose(results.y[:, 1], expected.y, rtol=1e-6)


def _raise(*args, **kwargs):
    raise AssertionError("daex must not use the legacy callbacks")


def test_daex_does_not_use_legacy_callbacks():
    params, t0, xy0 = make_problem()
    dae = def_semi_explicit_dae(derivative, constraint, params, t0, xy0)
    dae = dataclasses.replace(dae, **{name: _raise for name in LEGACY_CALLBACKS})
    ts = jnp.linspace(t0, 1.5, 3)

    def loss(params):
        return jnp.sum(daeint(params, dae, ts, xy0, options=OPTIONS).values.y)

    value, grad = jax.value_and_grad(loss)(params)
    assert np.isfinite(value)
    assert np.isfinite(grad.a)


def test_legacy_clear_cache_keeps_shared_cache():
    params, t0, xy0 = make_problem()
    dae = def_semi_explicit_dae(derivative, constraint, params, t0, xy0)
    daeint(params, dae, jnp.linspace(t0, 1.5, 3), xy0, options=OPTIONS)
    res_size = _res_fn._cache_size()
    jac_size = _jac_fn._cache_size()
    assert res_size > 0
    assert jac_size > 0

    dae._clear_cache()
    del dae
    gc.collect()

    assert _res_fn._cache_size() == res_size
    assert _jac_fn._cache_size() == jac_size


def test_with_statement_clears_legacy_cache():
    params, t0, xy0 = make_problem()
    calls = []
    dae = def_semi_explicit_dae(derivative, constraint, params, t0, xy0)
    dae = dataclasses.replace(dae, _clear_cache=lambda: calls.append(None))
    with dae:
        assert calls == []
    assert len(calls) == 1
