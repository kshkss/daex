from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest
from jax.flatten_util import ravel_pytree

from daex.semi_explicit import daeint, def_semi_explicit_dae

jax.config.update("jax_enable_x64", True)


class State(NamedTuple):
    x: jax.Array
    y: jax.Array


class Params(NamedTuple):
    a: jax.Array


def derivative(params: Params, t: jax.Array, stat: State) -> State:
    return State(
        x=None,
        y=params.a * stat.y,
    )


def constraint(params: Params, t: jax.Array, stat: State) -> jax.Array:
    return stat.x - stat.y**2 - t


def analytic(params: Params, ts: jax.Array, stat0: State) -> State:
    def result(params: Params, t: jax.Array, t0: jax.Array, stat0: State):
        y = stat0.y * jnp.exp(params.a * (t - t0))
        return State(x=y**2 + t, y=y)

    return jax.vmap(result, in_axes=(None, 0, None, None))(params, ts, ts[0], stat0)


@pytest.fixture
def params():
    return Params(a=jnp.array(0.7))


@pytest.fixture
def ts():
    return jnp.linspace(-0.5, 1.0, 11)


@pytest.fixture
def y0(ts):
    y_val = jnp.array(1.2)
    return State(y=y_val, x=y_val**2 + ts[0])


@pytest.fixture
def dae(params, y0):
    return def_semi_explicit_dae(derivative, constraint, params, jnp.array(0.0), y0)


@pytest.mark.parametrize("tangent_index", range(14))
def test_forward_mode_matches_analytic_gradient(dae, params, ts, y0, tangent_index):
    def loss(params, ts, y0):
        u, _ = daeint(params, dae, ts, y0, mode="forward")
        return u.y[-1]

    def loss_acc(params, ts, y0):
        u = analytic(params, ts, y0)
        return u.y[-1]

    primals = (params, ts, y0)
    flat, unravel = ravel_pytree(primals)
    tangents = unravel(jnp.zeros_like(flat).at[tangent_index].set(1.0))

    _, tangent_out = jax.jvp(loss, primals, tangents)
    grad_acc = jax.grad(loss_acc, argnums=[0, 1, 2])(params, ts, y0)
    grad_acc_flat, _ = ravel_pytree(grad_acc)

    assert jnp.allclose(tangent_out, grad_acc_flat[tangent_index], 1e-4, 1e-4)


@pytest.mark.parametrize("tangent_index", range(14))
def test_forward_over_forward_matches_analytic(dae, params, ts, y0, tangent_index):
    def loss(params, ts, y0):
        u, _ = daeint(params, dae, ts, y0, mode="forward")
        return u.y[-1]

    def loss_acc(params, ts, y0):
        u = analytic(params, ts, y0)
        return u.y[-1]

    primals = (params, ts, y0)
    flat, unravel = ravel_pytree(primals)
    tangents = unravel(jnp.zeros_like(flat).at[tangent_index].set(1.0))

    def directional_jvp(f):
        def g(primals):
            _, t_out = jax.jvp(f, primals, tangents)
            return t_out

        return g

    _, second_order = jax.jvp(directional_jvp(loss), (primals,), (tangents,))
    _, second_order_acc = jax.jvp(directional_jvp(loss_acc), (primals,), (tangents,))

    assert jnp.allclose(second_order, second_order_acc, 1e-4, 1e-4)
