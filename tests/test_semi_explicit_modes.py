import itertools
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


def test_invalid_mode_raises(dae, params, ts, y0):
    with pytest.raises(ValueError):
        daeint(params, dae, ts, y0, mode="bogus")


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


# Differentiation used at each order by each mode: R = reverse, F = forward.
MODE_ORDERS = {
    "forward": "FFF",
    "reverse": "RRR",
    "reverse_forward": "RFF",
    "alternating": "RFR",
}

# Every sequence of transformations up to third order, innermost first.
TRANSFORMS = [
    "".join(seq) for n in (1, 2, 3) for seq in itertools.product("RF", repeat=n)
]


# Errors with which JAX refuses a transformation the custom derivative rules do
# not support. Any other error is a bug.
UNSUPPORTED_AUTODIFF_ERRORS = [
    (TypeError, "can't apply forward-mode autodiff (jvp) to a custom_vjp function"),
    (ValueError, "Linearization failed to produce known values for all output primals"),
    (NotImplementedError, "Differentiation rule for 'custom_lin' not implemented"),
]


def _is_unsupported_autodiff_error(e):
    return any(
        isinstance(e, error_type) and message in str(e)
        for error_type, message in UNSUPPORTED_AUTODIFF_ERRORS
    )


def _tree_vdot(a, b):
    return sum(jnp.vdot(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


def _differentiate(f, transforms, unravel, size):
    """
    Apply `transforms` to the scalar function `f`, keeping the result scalar by
    taking a directional derivative at each order: R contracts the gradient
    with a direction, F takes a JVP along a direction.
    """
    for k, op in enumerate(transforms):
        direction = unravel(jnp.cos(0.7 * (k + 1) * (jnp.arange(size) + 1)))
        if op == "R":
            f = _grad_along(f, direction)
        else:
            f = _jvp_along(f, direction)
    return f


def _grad_along(f, direction):
    return lambda z: _tree_vdot(jax.grad(f)(z), direction)


def _jvp_along(f, direction):
    return lambda z: jax.jvp(f, (z,), (direction,))[1]


@pytest.mark.parametrize("transforms", TRANSFORMS)
@pytest.mark.parametrize("mode", MODE_ORDERS)
def test_mode_derivatives_match_analytic(dae, params, ts, y0, mode, transforms):
    """
    A sequence of transformations that follows the mode's differentiation
    order must succeed and match the analytic solution. Any other sequence is
    left to JAX: it may refuse the transformation as unsupported, but if it
    returns a value, the value must match.
    """

    def loss(z):
        params, ts, y0 = z
        u, _ = daeint(params, dae, ts, y0, mode=mode)
        return u.y[-1]

    def loss_acc(z):
        params, ts, y0 = z
        return analytic(params, ts, y0).y[-1]

    z = (params, ts, y0)
    flat, unravel = ravel_pytree(z)

    if MODE_ORDERS[mode].startswith(transforms):
        value = _differentiate(loss, transforms, unravel, flat.size)(z)
    else:
        try:
            value = _differentiate(loss, transforms, unravel, flat.size)(z)
        except Exception as e:
            if not _is_unsupported_autodiff_error(e):
                raise
            message = (str(e).splitlines() or [""])[0][:120]
            pytest.skip(f"Unsupported by JAX: {type(e).__name__}: {message}")

    expected = _differentiate(loss_acc, transforms, unravel, flat.size)(z)
    assert jnp.allclose(value, expected, 1e-4, 1e-4), (value, expected)
