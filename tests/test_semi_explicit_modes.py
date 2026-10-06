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
        y=stat.x * params.a * t,
    )


def constraint(params: Params, t: jax.Array, stat: State) -> jax.Array:
    return stat.x**2 - jnp.sum(stat.y)


def analytic(params: Params, ts: jax.Array, stat0: State) -> State:
    def result(params: Params, t: jax.Array, t0: jax.Array, stat0: State):
        x = 0.25 * params.a * (t**2 - t0**2) + jnp.sqrt(stat0.y)
        return State(x=x, y=x**2)

    return jax.vmap(result, in_axes=(None, 0, None, None))(params, ts, ts[0], stat0)


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


def assert_allclose_tree(a, b):
    matches = jax.tree.leaves(
        jax.tree.map(lambda x, y: jnp.allclose(x, y, 1e-4, 1e-4), a, b)
    )
    assert all(matches), (a, b)


def test_reverse_mode_matches_analytic_gradient(dae, params, ts, y0):
    def loss(params, ts, y0):
        u, _ = daeint(
            params,
            dae,
            ts,
            y0,
            mode="reverse",
            quad_order=9,
            options_adj={"calc_initcond": "yp0", "calc_init_dt": -0.01},
        )
        return u.y[-1]

    def loss_acc(params, ts, y0):
        u = analytic(params, ts, y0)
        return u.y[-1]

    grad = jax.grad(loss, argnums=[0, 1, 2])(params, ts, y0)
    grad_acc = jax.grad(loss_acc, argnums=[0, 1, 2])(params, ts, y0)
    assert_allclose_tree(grad, grad_acc)


def test_reverse_mode_gradient_includes_ts0_cotangent(dae, params, ts, y0):
    def loss(params, ts, y0):
        u, _ = daeint(
            params,
            dae,
            ts,
            y0,
            mode="reverse",
            quad_order=9,
            options_adj={"calc_initcond": "yp0", "calc_init_dt": -0.01},
        )
        return jnp.sum(u.y)

    grad_y0 = jax.grad(loss, argnums=2)(params, ts, y0)

    eps = 1e-6
    y0_plus = State(x=y0.x, y=y0.y + eps)
    y0_minus = State(x=y0.x, y=y0.y - eps)
    fd_y0 = (loss(params, ts, y0_plus) - loss(params, ts, y0_minus)) / (2 * eps)

    assert jnp.allclose(grad_y0.y, fd_y0, 1e-3, 1e-3)


def test_default_mode_is_reverse(dae, params, ts, y0):
    def loss_default(params, ts, y0):
        u, _ = daeint(params, dae, ts, y0)
        return u.y[-1]

    def loss_reverse(params, ts, y0):
        u, _ = daeint(params, dae, ts, y0, mode="reverse")
        return u.y[-1]

    grad_default = jax.grad(loss_default, argnums=[0, 1, 2])(params, ts, y0)
    grad_reverse = jax.grad(loss_reverse, argnums=[0, 1, 2])(params, ts, y0)
    assert_allclose_tree(grad_default, grad_reverse)


def test_reverse_forward_mode_first_order_matches_analytic_gradient(
    dae, params, ts, y0
):
    def loss(params, ts, y0):
        u, _ = daeint(
            params,
            dae,
            ts,
            y0,
            mode="reverse_forward",
            quad_order=9,
            options_adj={"calc_initcond": "yp0", "calc_init_dt": -0.01},
        )
        return u.y[-1]

    def loss_acc(params, ts, y0):
        u = analytic(params, ts, y0)
        return u.y[-1]

    grad = jax.grad(loss, argnums=[0, 1, 2])(params, ts, y0)
    grad_acc = jax.grad(loss_acc, argnums=[0, 1, 2])(params, ts, y0)
    assert_allclose_tree(grad, grad_acc)


@pytest.mark.parametrize("tangent_index", range(14))
def test_reverse_forward_mode_second_order_matches_analytic_hvp(
    dae, params, ts, y0, tangent_index
):
    def loss(params, ts, y0):
        u, _ = daeint(params, dae, ts, y0, mode="reverse_forward")
        return u.y[-1]

    def loss_acc(params, ts, y0):
        u = analytic(params, ts, y0)
        return u.y[-1]

    def hvp(f, primals, tangents):
        return jax.jvp(jax.grad(f, argnums=[0, 1, 2]), primals, tangents)[1]

    primals = (params, ts, y0)
    flat, unravel = ravel_pytree(primals)
    tangents = unravel(jnp.zeros_like(flat).at[tangent_index].set(1.0))

    hv = hvp(loss, primals, tangents)
    hv_acc = hvp(loss_acc, primals, tangents)
    assert_allclose_tree(hv, hv_acc)


def test_alternating_mode_first_order_matches_analytic_gradient(dae, params, ts, y0):
    def loss(params, ts, y0):
        u, _ = daeint(params, dae, ts, y0, mode="alternating")
        return u.y[-1]

    def loss_acc(params, ts, y0):
        u = analytic(params, ts, y0)
        return u.y[-1]

    grad = jax.grad(loss, argnums=[0, 1, 2])(params, ts, y0)
    grad_acc = jax.grad(loss_acc, argnums=[0, 1, 2])(params, ts, y0)
    assert_allclose_tree(grad, grad_acc)


@pytest.mark.parametrize("tangent_index", range(14))
def test_alternating_mode_second_order_matches_analytic_hvp(
    dae, params, ts, y0, tangent_index
):
    def loss(params, ts, y0):
        u, _ = daeint(params, dae, ts, y0, mode="alternating")
        return u.y[-1]

    def loss_acc(params, ts, y0):
        u = analytic(params, ts, y0)
        return u.y[-1]

    def hvp(f, primals, tangents):
        return jax.jvp(jax.grad(f, argnums=[0, 1, 2]), primals, tangents)[1]

    primals = (params, ts, y0)
    flat, unravel = ravel_pytree(primals)
    tangents = unravel(jnp.zeros_like(flat).at[tangent_index].set(1.0))

    hv = hvp(loss, primals, tangents)
    hv_acc = hvp(loss_acc, primals, tangents)
    assert_allclose_tree(hv, hv_acc)
