from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.ode import odeint

from daex.semi_explicit import daeint, def_semi_explicit_dae

jax.config.update("jax_enable_x64", True)


class State(NamedTuple):
    x: jax.Array  # algebraic, size 1
    y1: jax.Array  # differential
    y2: jax.Array  # differential


class Params(NamedTuple):
    a: jax.Array


def derivative(params: Params, t: jax.Array, stat: State) -> State:
    return State(x=None, y1=params.a * stat.y1, y2=params.a * stat.y2)


def constraint(params: Params, t: jax.Array, stat: State) -> jax.Array:
    return stat.x - stat.y1 - stat.y2


def analytic(params: Params, ts: jax.Array, stat0: State) -> State:
    def result(params: Params, t: jax.Array, t0: jax.Array, stat0: State):
        y1 = stat0.y1 * jnp.exp(params.a * (t - t0))
        y2 = stat0.y2 * jnp.exp(params.a * (t - t0))
        return State(x=y1 + y2, y1=y1, y2=y2)

    return jax.vmap(result, in_axes=(None, 0, None, None))(params, ts, ts[0], stat0)


@pytest.fixture
def params():
    return Params(a=jnp.array(0.7))


@pytest.fixture
def y0():
    return State(x=jnp.array(1.0), y1=jnp.array(0.3), y2=jnp.array(0.7))


@pytest.fixture
def dae(params, y0):
    return def_semi_explicit_dae(derivative, constraint, params, jnp.array(0.0), y0)


@pytest.fixture
def ts():
    return jnp.linspace(0.0, 1.0, 6)


def assert_allclose_tree(a, b):
    matches = jax.tree.leaves(
        jax.tree.map(lambda x, y: jnp.allclose(x, y, 1e-4, 1e-4), a, b)
    )
    assert all(matches), (a, b)


def test_reverse_mode_matches_analytic_gradient_with_unequal_sizes(dae, params, ts, y0):
    def loss(params, ts, y0):
        u, _ = daeint(params, dae, ts, y0)
        return u.y1[-1] + u.y2[-1]

    def loss_acc(params, ts, y0):
        u = analytic(params, ts, y0)
        return u.y1[-1] + u.y2[-1]

    grad = jax.grad(loss, argnums=[0, 1, 2])(params, ts, y0)
    grad_acc = jax.grad(loss_acc, argnums=[0, 1, 2])(params, ts, y0)
    assert_allclose_tree(grad, grad_acc)


def rotation_derivative(params: Params, t: jax.Array, stat: State) -> State:
    return State(x=None, y1=params.a * stat.y2, y2=-stat.y1)


def rotation_matrix(params: Params) -> jax.Array:
    # d(y1, y2)/dt = A @ (y1, y2); A is not symmetric, so dfdy^T != dfdy.
    return jnp.array([[0.0, params.a], [-1.0, 0.0]])


def rotation_analytic(params: Params, ts: jax.Array, stat0: State):
    A = rotation_matrix(params)
    y0 = jnp.stack([stat0.y1, stat0.y2])
    y = jax.vmap(lambda t: jax.scipy.linalg.expm(A * (t - ts[0])) @ y0)(ts)
    yp = y @ A.T
    values = State(x=y[:, 0] + y[:, 1], y1=y[:, 0], y2=y[:, 1])
    derivatives = State(x=yp[:, 0] + yp[:, 1], y1=yp[:, 0], y2=yp[:, 1])
    return values, derivatives


def test_reverse_mode_ts_gradient_with_nonsymmetric_jacobian(params, ts, y0):
    dae = def_semi_explicit_dae(
        rotation_derivative, constraint, params, jnp.array(0.0), y0
    )

    # Cotangents on the derivatives and the algebraic variable make the
    # dJ/dt terms depend on df/dy @ y' and dg/dy @ y'. The initial point is
    # excluded because cotangents on derivatives/x at ts[0] are not yet
    # propagated to y0.
    def reduce(u, up):
        return jnp.sum((up.y1 * ts)[1:]) + jnp.sum(up.y2[1:]) + jnp.sum(u.x[1:] ** 2)

    def loss(params, ts, y0):
        u, up = daeint(params, dae, ts, y0)
        return reduce(u, up)

    def loss_acc(params, ts, y0):
        u, up = rotation_analytic(params, ts, y0)
        return reduce(u, up)

    grad = jax.grad(loss, argnums=[0, 1, 2])(params, ts, y0)
    grad_acc = jax.grad(loss_acc, argnums=[0, 1, 2])(params, ts, y0)
    assert_allclose_tree(grad, grad_acc)


def forced_derivative(params: Params, t: jax.Array, stat: State) -> State:
    return State(x=None, y1=params.a * stat.y2 + t, y2=-stat.y1)


def forced_constraint(params: Params, t: jax.Array, stat: State) -> jax.Array:
    return stat.x - stat.y1 - stat.y2 - jnp.sin(t)


def forced_reference(params: Params, ts: jax.Array, stat0: State):
    def rhs(y, t, a):
        return jnp.stack([a * y[1] + t, -y[0]])

    y0 = jnp.stack([stat0.y1, stat0.y2])
    y = odeint(rhs, y0, ts, params.a, rtol=1e-12, atol=1e-12)
    yp = jax.vmap(rhs, in_axes=(0, 0, None))(y, ts, params.a)
    values = State(x=y[:, 0] + y[:, 1] + jnp.sin(ts), y1=y[:, 0], y2=y[:, 1])
    derivatives = State(x=yp[:, 0] + yp[:, 1] + jnp.cos(ts), y1=yp[:, 0], y2=yp[:, 1])
    return values, derivatives


TIGHT = dict(rtol=1e-10, atol=1e-12)


def test_reverse_mode_gradient_with_time_dependent_system(params, ts, y0):
    # f and g depend on t explicitly, so df/dt and dg/dt enter dJ/dt.
    dae = def_semi_explicit_dae(
        forced_derivative, forced_constraint, params, jnp.array(0.0), y0
    )

    # The initial point is excluded for the same reason as above.
    def reduce(u, up):
        return jnp.sum((up.y1 * ts)[1:]) + jnp.sum(up.y2[1:]) + jnp.sum(u.x[1:] ** 2)

    def loss(params, ts, y0):
        u, up = daeint(params, dae, ts, y0, options=TIGHT, options_adj=TIGHT)
        return reduce(u, up)

    def loss_ref(params, ts, y0):
        u, up = forced_reference(params, ts, y0)
        return reduce(u, up)

    grad = jax.grad(loss, argnums=[0, 1, 2])(params, ts, y0)
    grad_ref = jax.grad(loss_ref, argnums=[0, 1, 2])(params, ts, y0)
    assert_allclose_tree(grad, grad_ref)
