from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.ode import odeint

from daex.semi_explicit import Results, adjoint, daeint, def_semi_explicit_dae

jax.config.update("jax_enable_x64", True)

# Two algebraic variables with nonsymmetric dg/dx and df/dx, so that a
# transposed Jacobian or a mixed-up [x, lambda_g, lambda_f] layout in the
# adjoint DAE changes the result.
A = jnp.array([[2.0, 1.0], [0.5, 3.0]])  # dg/dx


class State(NamedTuple):
    x: jax.Array  # algebraic, shape (2,)
    y: jax.Array  # differential, shape (2,)


class Params(NamedTuple):
    a: jax.Array


def derivative(params: Params, t: jax.Array, stat: State) -> State:
    x, y = stat.x, stat.y
    return State(
        x=None,
        y=jnp.stack(
            [params.a * x[0] + 0.1 * x[1] - 0.3 * y[1], -x[1] * y[0] + 0.2 * t]
        ),
    )


def g_rhs(t, y):
    return jnp.stack([y[0] + 0.5 * jnp.sin(t), y[1] ** 2])


def constraint(params: Params, t: jax.Array, stat: State) -> jax.Array:
    return A @ stat.x - g_rhs(t, stat.y)


def solve_x(t, y):
    return jnp.linalg.solve(A, g_rhs(t, y))


def reference(params: Params, ts: jax.Array, stat0: State):
    def rhs(y, t, a):
        return derivative(Params(a=a), t, State(x=solve_x(t, y), y=y)).y

    y = odeint(rhs, stat0.y, ts, params.a, rtol=1e-12, atol=1e-12)
    yp = jax.vmap(rhs, in_axes=(0, 0, None))(y, ts, params.a)
    x = jax.vmap(solve_x)(ts, y)
    xp = jax.vmap(lambda t, y, yp: jax.jvp(solve_x, (t, y), (1.0, yp))[1])(ts, y, yp)
    return State(x=x, y=y), State(x=xp, y=yp)


TIGHT = dict(rtol=1e-10, atol=1e-12)
WX = jnp.array([0.7, -1.3])
WY = jnp.array([0.4, 0.9])
WYP = jnp.array([-0.6, 1.1])


@pytest.fixture
def params():
    return Params(a=jnp.array(0.8))


@pytest.fixture
def y0():
    y = jnp.array([0.5, 0.4])
    return State(x=solve_x(jnp.array(0.0), y), y=y)


@pytest.fixture
def dae(params, y0):
    return def_semi_explicit_dae(derivative, constraint, params, jnp.array(0.0), y0)


@pytest.fixture
def ts():
    return jnp.linspace(0.0, 1.0, 5)


def assert_allclose_tree(a, b):
    matches = jax.tree.leaves(
        jax.tree.map(lambda x, y: jnp.allclose(x, y, 1e-5, 1e-5), a, b)
    )
    assert all(matches), (a, b)


def test_reverse_mode_gradient_with_two_algebraic_variables(dae, params, ts, y0):
    # Includes ts[0], so the point multipliers there go through dg/dx^T too.
    def reduce(u, up):
        return (
            jnp.sum((u.x * ts[:, None]) @ WX)
            + jnp.sum(u.y @ WY)
            + jnp.sum(up.y @ WYP)
            + jnp.sum(up.x[:, 0] * up.x[:, 1])
        )

    def loss(params, ts, y0):
        u, up = daeint(params, dae, ts, y0, options=TIGHT, options_adj=TIGHT)
        return reduce(u, up)

    def loss_ref(params, ts, y0):
        u, up = reference(params, ts, y0)
        return reduce(u, up)

    grad = jax.grad(loss, argnums=[0, 1, 2])(params, ts, y0)
    grad_ref = jax.grad(loss_ref, argnums=[0, 1, 2])(params, ts, y0)
    # y0.x is not an independent input (x0 is determined by y0).
    assert_allclose_tree(grad[:2], grad_ref[:2])
    assert_allclose_tree(grad[2].y, grad_ref[2].y)


def test_adjoint_with_two_algebraic_variables(dae, params, ts, y0):
    result = daeint(params, dae, ts, y0, options=TIGHT, options_adj=TIGHT)

    # Cotangents on x, y and y' at every point except ts[0], so that
    # initial_value = mu_{y_0} equals dJ/dy0.
    mask = (jnp.arange(ts.size) > 0)[:, None]
    wx = mask * WX
    wy = mask * WY
    wyp = mask * WYP
    cotangent = Results(
        values=State(x=wx, y=wy),
        derivatives=State(x=jnp.zeros_like(wx), y=wyp),
    )

    def loss_ref(y):
        u, up = reference(params, ts, State(x=solve_x(ts[0], y), y=y))
        return jnp.sum(u.x * wx) + jnp.sum(u.y * wy) + jnp.sum(up.y * wyp)

    out = adjoint(params, dae, ts, result, cotangent, options=TIGHT)

    assert jnp.allclose(out.initial_value.y, jax.grad(loss_ref)(y0.y), 1e-5, 1e-5)

    # constraint = lambda_g satisfies dg/dx^T lambda_g = df/dx^T lambda_f at
    # both ends of every interval. Build the dense Jacobians here from the
    # user-level functions, independently of the implementation's vjp.
    def dfdx(t, x, y):
        return jax.jacfwd(lambda x: derivative(params, t, State(x=x, y=y)).y)(x)

    for k in range(ts.size - 1):
        for side in (0, 1):
            i = k + side
            x_i = result.values.x[i]
            y_i = result.values.y[i]
            lam = out.derivative.y[k, side]
            lam_g = out.constraint.x[k, side]
            assert jnp.allclose(
                A.T @ lam_g, dfdx(ts[i], x_i, y_i).T @ lam, 1e-10, 1e-10
            )
            # With nonsymmetric A, using A instead of A^T would not match.
            assert not jnp.allclose(A @ lam_g, dfdx(ts[i], x_i, y_i).T @ lam)
