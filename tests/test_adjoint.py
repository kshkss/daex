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
        u, _ = daeint(params, dae, ts, y0, mode="reverse", **DAEINT_OPTIONS)
        return jnp.sum(u.y)

    result = daeint(params, dae, ts, y0, mode="reverse", **DAEINT_OPTIONS)
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
        u, _ = daeint(params, dae, ts, y0, mode="reverse", **DAEINT_OPTIONS)
        return u.y[-1]

    result = daeint(params, dae, ts, y0, mode="reverse", **DAEINT_OPTIONS)
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
    result = daeint(params, dae, ts, y0, mode="reverse", **DAEINT_OPTIONS)
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

    # adjoint() now returns, for each of the len(ts)-1 forward integration
    # intervals, the continuous adjoint's value at BOTH endpoints (the two
    # one-sided limits bounding that interval), instead of a single
    # C0-continuous value per ts point. derivative[k, 0] = lambda(ts[k]^+)
    # (pre-jump, entering interval k) and derivative[k, 1] = lambda(ts[k+1]^-)
    # (post-jump, the value that seeds the next interval backward).
    #
    # In this test's cotangent setup (nonzero only at the LAST point), ts[0]
    # has zero own cotangent contribution, so lambda(ts[0]^+) == lambda(ts[0]^-)
    # here specifically -- i.e. derivative[0, 0] genuinely equals
    # initial_value (lambda(ts[0]^-)), not just by relabeling.
    assert jnp.allclose(out.initial_value.y, out.derivative.y[0, 0])

    # mu(t) should be recoverable purely from lam(t) and the forward solution,
    # by re-deriving the same algebraic relation used inside deriv_adj/da_fn.
    # Check this comprehensively across every interval endpoint (both sides
    # of every one of the len(ts)-1 intervals), not just a single point.
    a, _ = ravel_pytree(params)
    points = ts.shape[0]

    def ravel_state_leaf(leaf):
        x1, _ = ravel_pytree(dae.partition(leaf)[0])
        y1, _ = ravel_pytree(dae.partition(leaf)[1])
        return x1, y1

    def expected_mu_at(k, side):
        # Interval k covers [ts[k], ts[k+1]]; side 0 -> ts[k], side 1 -> ts[k+1].
        i = k + side
        t = ts[i]
        x1, y1 = ravel_state_leaf(jax.tree.map(lambda leaf: leaf[i], result.values))
        lam1, _ = ravel_pytree(
            dae.partition(
                jax.tree.map(lambda leaf: leaf[k, side], out.derivative)
            )[1]
        )
        _, vjp_deriv = jax.vjp(dae.deriv_fn, a, t, x1, y1)
        _, _, zdfdx, _ = vjp_deriv(lam1)
        dgdx = jax.jacfwd(dae.const_fn, argnums=2)(a, t, x1, y1)
        return jnp.linalg.solve(dgdx.T, zdfdx)

    def actual_mu_at(k, side):
        mu1, _ = ravel_pytree(
            dae.partition(jax.tree.map(lambda leaf: leaf[k, side], out.constraint))[0]
        )
        return mu1

    for k in range(points - 1):
        for side in (0, 1):
            assert jnp.allclose(actual_mu_at(k, side), expected_mu_at(k, side))


def test_adjoint_matches_finite_difference_with_interior_cotangent(dae, params, ts, y0):
    def loss(params, ts, y0):
        u, _ = daeint(params, dae, ts, y0, mode="reverse", **DAEINT_OPTIONS)
        return u.y[5]

    result = daeint(params, dae, ts, y0, mode="reverse", **DAEINT_OPTIONS)
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

    # The cotangent is injected only at ts[5], so the continuous adjoint
    # must have a genuine jump discontinuity exactly there, of size equal
    # to that point's own cotangent contribution. A regression that
    # duplicated the post-jump value into both sides (discarding
    # lambda(ts[k]+)) would make this difference zero instead.
    jump = z.derivative.y[4, 1] - z.derivative.y[5, 0]
    assert jnp.allclose(jump, cotangent.y[5], 1e-6, 1e-6)

    # mu is recomputed pointwise from lambda, so validate BOTH sides at
    # ts[5] independently against the manual algebraic formula (reused
    # from test_adjoint_mu_is_recomputed_from_lambda), and confirm they
    # differ -- i.e. the constraint sides carry the same jump information,
    # not just the costate.
    a, _ = ravel_pytree(params)
    t5 = ts[5]
    x5, _ = ravel_pytree(
        dae.partition(jax.tree.map(lambda leaf: leaf[5], result.values))[0]
    )
    y5, _ = ravel_pytree(
        dae.partition(jax.tree.map(lambda leaf: leaf[5], result.values))[1]
    )

    def expected_mu(lam_var):
        lam1, _ = ravel_pytree(dae.partition(lam_var)[1])
        _, vjp_deriv = jax.vjp(dae.deriv_fn, a, t5, x5, y5)
        _, _, zdfdx, _ = vjp_deriv(lam1)
        dgdx = jax.jacfwd(dae.const_fn, argnums=2)(a, t5, x5, y5)
        return jnp.linalg.solve(dgdx.T, zdfdx)

    def actual_mu(constraint_var):
        mu1, _ = ravel_pytree(dae.partition(constraint_var)[0])
        return mu1

    lam_pre_5 = jax.tree.map(lambda leaf: leaf[5, 0], z.derivative)
    lam_post_5 = jax.tree.map(lambda leaf: leaf[4, 1], z.derivative)
    mu_pre_5 = jax.tree.map(lambda leaf: leaf[5, 0], z.constraint)
    mu_post_5 = jax.tree.map(lambda leaf: leaf[4, 1], z.constraint)

    assert jnp.allclose(actual_mu(mu_pre_5), expected_mu(lam_pre_5))
    assert jnp.allclose(actual_mu(mu_post_5), expected_mu(lam_post_5))
    assert not jnp.allclose(actual_mu(mu_pre_5), actual_mu(mu_post_5))
