from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from daex.semi_explicit import _make_model, def_semi_explicit_dae, _forward_mode

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


@pytest.fixture(params=[11, 2], ids=["11pts", "2pts"])
def problem(request):
    params = Params(a=jnp.array(1.0))
    xy0 = State(y=jnp.array(2.0), x=jnp.sqrt(2.0))
    dae = def_semi_explicit_dae(derivative, constraint, params, jnp.array(0.0), xy0)
    x0, y0 = dae.partition(xy0)
    model, _, y, a = _make_model(dae, x0, y0, params)
    ts = jnp.linspace(-0.5, 1.0, request.param)

    def forward(a, ts, y):
        # x0 = sqrt(y0) keeps the initial condition consistent with constraint.
        return _forward_mode(model, a, ts, jnp.sqrt(y), y, {})

    return forward, a, ts, y


@pytest.mark.parametrize("direction", ["params", "ts", "y0"])
def test_run_forward_jvp_matches_finite_difference(problem, direction):
    forward, a, ts, y = problem
    tangents = {
        "params": (jnp.ones_like(a), jnp.zeros_like(ts), jnp.zeros_like(y)),
        "ts": (
            jnp.zeros_like(a),
            jnp.zeros_like(ts).at[0].set(1.0).at[-1].set(0.5),
            jnp.zeros_like(y),
        ),
        "y0": (jnp.zeros_like(a), jnp.zeros_like(ts), jnp.ones_like(y)),
    }[direction]

    _, (dx, dy, dyp) = jax.jvp(forward, (a, ts, y), tangents)

    eps = 1e-6
    plus = forward(*(p + eps * t for p, t in zip((a, ts, y), tangents)))
    minus = forward(*(p - eps * t for p, t in zip((a, ts, y), tangents)))
    fd_x, fd_y, fd_yp = ((p - m) / (2 * eps) for p, m in zip(plus, minus))

    assert jnp.allclose(dx, fd_x, 1e-3, 1e-3)
    assert jnp.allclose(dy, fd_y, 1e-3, 1e-3)
    assert jnp.allclose(dyp, fd_yp, 1e-2, 1e-2)


def test_run_forward_yp_matches_derivative(problem):
    forward, a, ts, y = problem
    x, y, yp = forward(a, ts, y)
    expected = jax.vmap(lambda t, x, y: derivative(Params(a[0]), t, State(x, y)).y)(
        ts, x[:, 0], y[:, 0]
    )
    assert jnp.allclose(yp[:, 0], expected, 1e-3, 1e-3)
