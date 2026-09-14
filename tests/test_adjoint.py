from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from jax.flatten_util import ravel_pytree

from daex.semi_explicit import Results, adjoint, daeint, def_semi_explicit_dae

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


DAEINT_OPTIONS = dict(
    quad_order=9,
    options_adj={"calc_initcond": "yp0", "calc_init_dt": -0.01},
)


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


def finite_diff_grad_y(loss, params, ts, y0, eps=1e-6):
    y0_plus = State(x=y0.x, y=y0.y + eps)
    y0_minus = State(x=y0.x, y=y0.y - eps)
    return (loss(params, ts, y0_plus) - loss(params, ts, y0_minus)) / (2 * eps)


def test_adjoint_matches_finite_difference_with_cotangent_at_every_point(
    dae, params, ts, y0
):
    def loss(params, ts, y0):
        u, _ = daeint(params, dae, ts, y0, **DAEINT_OPTIONS)
        return jnp.sum(u.y)

    result = daeint(params, dae, ts, y0, **DAEINT_OPTIONS)
    cotangent = State(
        x=jnp.zeros_like(result.values.x), y=jnp.ones_like(result.values.y)
    )
    cotangent_derivative = State(
        x=jnp.zeros_like(result.derivatives.x),
        y=jnp.zeros_like(result.derivatives.y),
    )

    z = adjoint(
        params,
        dae,
        ts,
        result,
        Results(values=cotangent, derivatives=cotangent_derivative),
    )

    fd = finite_diff_grad_y(loss, params, ts, y0)
    assert jnp.allclose(z.initial_value.y, fd, 1e-3, 1e-3)


def test_adjoint_matches_finite_difference_with_terminal_cotangent(dae, params, ts, y0):
    def loss(params, ts, y0):
        u, _ = daeint(params, dae, ts, y0, **DAEINT_OPTIONS)
        return u.y[-1]

    result = daeint(params, dae, ts, y0, **DAEINT_OPTIONS)
    cotangent = State(
        x=jnp.zeros_like(result.values.x),
        y=jnp.zeros_like(result.values.y).at[-1].set(1.0),
    )
    cotangent_derivative = State(
        x=jnp.zeros_like(result.derivatives.x),
        y=jnp.zeros_like(result.derivatives.y),
    )

    z = adjoint(
        params,
        dae,
        ts,
        result,
        Results(values=cotangent, derivatives=cotangent_derivative),
    )

    fd = finite_diff_grad_y(loss, params, ts, y0)
    assert jnp.allclose(z.initial_value.y, fd, 1e-3, 1e-3)


def test_adjoint_mu_is_recomputed_from_lambda(dae, params, ts, y0):
    result = daeint(params, dae, ts, y0, **DAEINT_OPTIONS)
    cotangent = State(
        x=jnp.zeros_like(result.values.x),
        y=jnp.zeros_like(result.values.y).at[-1].set(1.0),
    )
    cotangent_derivative = State(
        x=jnp.zeros_like(result.derivatives.x),
        y=jnp.zeros_like(result.derivatives.y),
    )

    out = adjoint(
        params,
        dae,
        ts,
        result,
        Results(values=cotangent, derivatives=cotangent_derivative),
    )

    # lam only populates the differential (y) part; mu only the algebraic (x) part.
    assert out.derivative.x is None
    assert jnp.allclose(out.initial_value.y, out.derivative.y[0])

    # mu(t) should be recoverable purely from lam(t) and the forward solution,
    # by re-deriving the same algebraic relation used inside deriv_adj/da_fn.
    a, _ = ravel_pytree(params)
    i = -1
    t = ts[i]
    x1, _ = ravel_pytree(
        dae.partition(jax.tree.map(lambda leaf: leaf[i], result.values))[0]
    )
    y1, _ = ravel_pytree(
        dae.partition(jax.tree.map(lambda leaf: leaf[i], result.values))[1]
    )
    lam1, _ = ravel_pytree(
        dae.partition(jax.tree.map(lambda leaf: leaf[i], out.derivative))[1]
    )

    _, vjp_deriv = jax.vjp(dae.deriv_fn, a, t, x1, y1)
    _, _, zdfdx, _ = vjp_deriv(lam1)
    dgdx = jax.jacfwd(dae.const_fn, argnums=2)(a, t, x1, y1)
    expected_mu, _ = ravel_pytree(
        dae.partition(jax.tree.map(lambda leaf: leaf[i], out.constraint))[0]
    )
    assert jnp.allclose(expected_mu, jnp.linalg.solve(dgdx.T, zdfdx))


def test_adjoint_matches_finite_difference_with_interior_cotangent(dae, params, ts, y0):
    def loss(params, ts, y0):
        u, _ = daeint(params, dae, ts, y0, **DAEINT_OPTIONS)
        return u.y[5]

    result = daeint(params, dae, ts, y0, **DAEINT_OPTIONS)
    cotangent = State(
        x=jnp.zeros_like(result.values.x),
        y=jnp.zeros_like(result.values.y).at[5].set(1.0),
    )
    cotangent_derivative = State(
        x=jnp.zeros_like(result.derivatives.x),
        y=jnp.zeros_like(result.derivatives.y),
    )

    z = adjoint(
        params,
        dae,
        ts,
        result,
        Results(values=cotangent, derivatives=cotangent_derivative),
    )

    fd = finite_diff_grad_y(loss, params, ts, y0)
    assert jnp.allclose(z.initial_value.y, fd, 1e-3, 1e-3)
