from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from jax.flatten_util import ravel_pytree

from daex import utils
from daex.semi_explicit import (
    Results,
    _bwd_int,
    _daeint_bwd_step2,
    _make_model,
    adjoint,
    daeint,
    def_semi_explicit_dae,
)

jax.config.update("jax_enable_x64", True)


class State(NamedTuple):
    x: jax.Array
    y: jax.Array


class Params(NamedTuple):
    a: jax.Array


def derivative(params: Params, t: jax.Array, stat: State) -> State:
    return State(
        x=None,
        y=stat.x * params.a * t,
    )


def constraint(params: Params, t: jax.Array, stat: State) -> jax.Array:
    return stat.x**2 - jnp.sum(stat.y)


@pytest.fixture
def params():
    return Params(a=jnp.array(1.0))


@pytest.fixture
def y0():
    return State(y=jnp.array(2.0), x=jnp.sqrt(2.0))


@pytest.fixture
def dae(params, y0):
    return def_semi_explicit_dae(derivative, constraint, params, jnp.array(0.0), y0)


@pytest.fixture
def ts():
    return jnp.linspace(-0.5, 1.0, 11)


def ravel_solution(dae, solution: Results):
    def ravel_xy(xy):
        x, y = dae.partition(xy)
        return ravel_pytree(x)[0], ravel_pytree(y)[0]

    x, y = jax.vmap(ravel_xy)(solution.values)
    _, yp = jax.vmap(ravel_xy)(solution.derivatives)
    return x, y, yp


def test_bwd_int_matches_adjoint_on_each_interval(dae, params, ts, y0):
    result = daeint(params, dae, ts, y0)
    cotangent = Results(
        values=State(
            x=jnp.zeros_like(result.values.x), y=jnp.ones_like(result.values.y)
        ),
        derivatives=State(
            x=jnp.zeros_like(result.derivatives.x),
            y=jnp.zeros_like(result.derivatives.y),
        ),
    )
    z = adjoint(params, dae, ts, result, cotangent)

    x0, y0_ = dae.partition(y0)
    model, _, _, a = _make_model(dae, x0, y0_, params)
    x, y, yp = ravel_solution(dae, result)
    lam_f = jax.vmap(jax.vmap(lambda v: ravel_pytree(v)[0]))(z.lambda_f)
    lam_g = jax.vmap(jax.vmap(lambda v: ravel_pytree(v)[0]))(z.lambda_g)

    for k in range(ts.shape[0] - 1):
        xy = _bwd_int(
            model,
            {},
            a,
            ts[k : k + 2],
            x[k : k + 2],
            y[k : k + 2],
            yp[k : k + 2],
            lam_f[k, 1],
            quad_order=5,
            mode="forward",
        )
        assert jnp.allclose(xy.lam_f, lam_f[k])
        assert jnp.allclose(xy.lam_g, lam_g[k])


def test_bwd_int_matches_daeint_bwd_step2(dae, params, y0):
    quad_order = 9
    n = (quad_order + 3) // 2
    ts, ws = utils.divide_intervals(jnp.array([-0.5]), jnp.array([1.0]), n=n)
    result = daeint(params, dae, ts, y0)

    x0, y0_ = dae.partition(y0)
    model, _, _, a = _make_model(dae, x0, y0_, params)
    x, y, yp = ravel_solution(dae, result)
    lam_f1 = jnp.ones_like(y[-1])

    xy = _bwd_int(
        model,
        {},
        a,
        ts,
        x,
        y,
        yp,
        lam_f1,
        quad_order=quad_order,
        mode="reverse_forward",
    )
    lam_f0, _ = _daeint_bwd_step2(
        model,
        {},
        a,
        ts,
        ws[0],
        x,
        y,
        yp,
        lam_f1,
        quad_order=quad_order,
        mode="reverse_forward",
    )

    assert xy.lam_f.shape == (n, y.shape[1])
    assert xy.lam_g.shape == (n, x.shape[1])
    assert jnp.allclose(xy.lam_f[0], lam_f0)
    assert jnp.allclose(xy.lam_f[-1], lam_f1)
