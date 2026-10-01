from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.ode import odeint

from daex.semi_explicit import Results, adjoint, daeint, def_semi_explicit_dae

jax.config.update("jax_enable_x64", True)

# Two algebraic variables with nonsymmetric dg/dx and df/dx, so that a
# transposed Jacobian or a mixed-up [x, lambda_g, lambda_f] layout in the
# adjoint DAE changes the result. The constraint depends on the parameter b
# (and f only on a), so the mu_g^T dg/da and lambda_g^T dg/da terms of the
# parameter gradient are exercised separately from the df/da terms.
A = jnp.array([[2.0, 1.0], [0.5, 3.0]])  # dg/dx


class State(NamedTuple):
    x: jax.Array  # algebraic, shape (2,)
    y: jax.Array  # differential, shape (2,)


class Params(NamedTuple):
    a: jax.Array  # enters f only
    b: jax.Array  # enters g only


def derivative(params: Params, t: jax.Array, stat: State) -> State:
    x, y = stat.x, stat.y
    return State(
        x=None,
        y=jnp.stack(
            [params.a * x[0] + 0.1 * x[1] - 0.3 * y[1], -x[1] * y[0] + 0.2 * t]
        ),
    )


def g_rhs(params: Params, t, y):
    # dg/db is nonzero at every point, including t = 0.
    return jnp.stack(
        [
            y[0] + params.b * jnp.sin(t) + 0.3 * params.b,
            y[1] ** 2 + params.b * y[0],
        ]
    )


def constraint(params: Params, t: jax.Array, stat: State) -> jax.Array:
    return A @ stat.x - g_rhs(params, t, stat.y)


def solve_x(params: Params, t, y):
    return jnp.linalg.solve(A, g_rhs(params, t, y))


def reference(params: Params, ts: jax.Array, stat0: State):
    def rhs(y, t, params):
        return derivative(params, t, State(x=solve_x(params, t, y), y=y)).y

    y = odeint(rhs, stat0.y, ts, params, rtol=1e-12, atol=1e-12)
    yp = jax.vmap(rhs, in_axes=(0, 0, None))(y, ts, params)
    x = jax.vmap(solve_x, in_axes=(None, 0, 0))(params, ts, y)
    xp = jax.vmap(
        lambda t, y, yp: jax.jvp(lambda t, y: solve_x(params, t, y), (t, y), (1.0, yp))[
            1
        ]
    )(ts, y, yp)
    return State(x=x, y=y), State(x=xp, y=yp)


TIGHT = dict(rtol=1e-10, atol=1e-12)
WX = jnp.array([0.7, -1.3])
WY = jnp.array([0.4, 0.9])
WYP = jnp.array([-0.6, 1.1])


@pytest.fixture
def params():
    return Params(a=jnp.array(0.8), b=jnp.array(0.6))


@pytest.fixture
def y0(params):
    y = jnp.array([0.5, 0.4])
    return State(x=solve_x(params, jnp.array(0.0), y), y=y)


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


def reduce(ts, u, up):
    # Includes ts[0], so the point multipliers there go through dg/dx^T too.
    return (
        jnp.sum((u.x * ts[:, None]) @ WX)
        + jnp.sum(u.y @ WY)
        + jnp.sum(up.y @ WYP)
        + jnp.sum(up.x[:, 0] * up.x[:, 1])
    )


def test_reverse_mode_gradient_with_two_algebraic_variables(dae, params, ts, y0):
    def loss(params, ts, y0):
        u, up = daeint(params, dae, ts, y0, options=TIGHT, options_adj=TIGHT)
        return reduce(ts, u, up)

    def loss_ref(params, ts, y0):
        u, up = reference(params, ts, y0)
        return reduce(ts, u, up)

    grad = jax.grad(loss, argnums=[0, 1, 2])(params, ts, y0)
    grad_ref = jax.grad(loss_ref, argnums=[0, 1, 2])(params, ts, y0)
    # The dg/db path carries a nonzero gradient.
    assert jnp.abs(grad_ref[0].b) > 1e-2
    # y0.x is not an independent input (x0 is determined by y0).
    assert_allclose_tree(grad[:2], grad_ref[:2])
    assert_allclose_tree(grad[2].y, grad_ref[2].y)


def test_reverse_mode_matches_finite_difference_with_two_algebraic_variables(
    dae, params, ts, y0
):
    # Central differences of daeint itself, independent of any AD rule.
    def loss(params, ts, y):
        y0 = State(x=solve_x(params, ts[0], y), y=y)
        u, up = daeint(params, dae, ts, y0, options=TIGHT, options_adj=TIGHT)
        return reduce(ts, u, up)

    grad_params, grad_ts, grad_y = jax.grad(loss, argnums=[0, 1, 2])(params, ts, y0.y)

    eps = 1e-4

    def central(direction):
        d_params, d_ts, d_y = direction
        plus = loss(
            jax.tree.map(lambda p, d: p + eps * d, params, d_params),
            ts + eps * d_ts,
            y0.y + eps * d_y,
        )
        minus = loss(
            jax.tree.map(lambda p, d: p - eps * d, params, d_params),
            ts - eps * d_ts,
            y0.y - eps * d_y,
        )
        return (plus - minus) / (2 * eps)

    zero_params = Params(a=jnp.array(0.0), b=jnp.array(0.0))
    zero_ts = jnp.zeros_like(ts)
    zero_y = jnp.zeros_like(y0.y)
    cases = {
        "a": ((zero_params._replace(a=jnp.array(1.0)), zero_ts, zero_y), grad_params.a),
        "b": ((zero_params._replace(b=jnp.array(1.0)), zero_ts, zero_y), grad_params.b),
        "y0[0]": ((zero_params, zero_ts, zero_y.at[0].set(1.0)), grad_y[0]),
        "y0[1]": ((zero_params, zero_ts, zero_y.at[1].set(1.0)), grad_y[1]),
        "ts[2]": ((zero_params, zero_ts.at[2].set(1.0), zero_y), grad_ts[2]),
    }
    for name, (direction, expected) in cases.items():
        fd = central(direction)
        assert jnp.allclose(expected, fd, 1e-6, 1e-6), (name, expected, fd)

    # A generic direction mixing all inputs.
    key_a, key_t, key_y = jax.random.split(jax.random.PRNGKey(0), 3)
    d_params = Params(a=jax.random.normal(key_a), b=-jax.random.normal(key_a))
    d_ts = 0.1 * jax.random.normal(key_t, ts.shape)
    d_y = jax.random.normal(key_y, y0.y.shape)
    directional = (
        grad_params.a * d_params.a
        + grad_params.b * d_params.b
        + jnp.dot(grad_ts, d_ts)
        + jnp.dot(grad_y, d_y)
    )
    fd = central((d_params, d_ts, d_y))
    assert jnp.allclose(directional, fd, 1e-6, 1e-6), (directional, fd)


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
        u, up = reference(params, ts, State(x=solve_x(params, ts[0], y), y=y))
        return jnp.sum(u.x * wx) + jnp.sum(u.y * wy) + jnp.sum(up.y * wyp)

    out = adjoint(params, dae, ts, result, cotangent, options=TIGHT)

    assert jnp.allclose(out.initial_value.y, jax.grad(loss_ref)(y0.y), 1e-5, 1e-5)

    # Build the dense Jacobians here from the user-level functions,
    # independently of the implementation's vjp.
    def dfdx(t, x, y):
        return jax.jacfwd(lambda x: derivative(params, t, State(x=x, y=y)).y)(x)

    # -lambda_f at a point is the sensitivity of the loss over the later
    # points to the state there: lambda_f(ts[k]+) excludes the cotangents at
    # ts[k] itself, and lambda_f(ts[k]-) includes them.
    def future_loss(y, k, include_own):
        own = jnp.where(include_own, 1.0, 0.0)
        weight = jnp.ones(ts.size - k).at[0].set(own)[:, None]
        u, up = reference(params, ts[k:], State(x=solve_x(params, ts[k], y), y=y))
        return (
            jnp.sum(u.x * wx[k:] * weight)
            + jnp.sum(u.y * wy[k:] * weight)
            + jnp.sum(up.y * wyp[k:] * weight)
        )

    u_ref, _ = reference(params, ts, y0)

    for k in range(ts.size - 1):
        for side in (0, 1):
            i = k + side
            # side 0: lambda_f(ts[k]+), side 1: lambda_f(ts[k+1]-)
            lam_ref = -jax.grad(future_loss)(u_ref.y[i], i, side == 1)
            lam_g_ref = jnp.linalg.solve(
                A.T, dfdx(ts[i], u_ref.x[i], u_ref.y[i]).T @ lam_ref
            )
            lam = out.derivative.y[k, side]
            lam_g = out.constraint.x[k, side]
            assert jnp.allclose(lam, lam_ref, 1e-5, 1e-5), (k, side, lam, lam_ref)
            assert jnp.allclose(lam_g, lam_g_ref, 1e-5, 1e-5), (
                k,
                side,
                lam_g,
                lam_g_ref,
            )

            # constraint = lambda_g satisfies dg/dx^T lambda_g = df/dx^T
            # lambda_f, and with nonsymmetric A, using A instead of A^T would
            # not match.
            x_i = result.values.x[i]
            y_i = result.values.y[i]
            assert jnp.allclose(
                A.T @ lam_g, dfdx(ts[i], x_i, y_i).T @ lam, 1e-10, 1e-10
            )
            assert not jnp.allclose(A @ lam_g, dfdx(ts[i], x_i, y_i).T @ lam)
