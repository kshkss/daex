import jax
import jax.numpy as jnp

from daex.utils import HermiteSpline

jax.config.update("jax_enable_x64", True)


def f(t):
    return jnp.sin(3 * t)


def df(t):
    return 3 * jnp.cos(3 * t)


def test_hermite_spline_descending_points_match_nodes():
    points = jnp.array([-0.35, -0.3555, -0.386, -0.3915])
    spline = HermiteSpline(points, f(points), df(points))
    assert jnp.allclose(jax.vmap(spline)(points), f(points), 0.0, 1e-14)


def test_hermite_spline_descending_points_match_ascending():
    points = jnp.array([-0.3915, -0.386, -0.3555, -0.35])
    asc = HermiteSpline(points, f(points), df(points))
    desc = HermiteSpline(points[::-1], f(points[::-1]), df(points[::-1]))
    ts = jnp.linspace(points[0], points[-1], 17)
    assert jnp.allclose(jax.vmap(desc)(ts), jax.vmap(asc)(ts), 0.0, 1e-14)
