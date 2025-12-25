import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any
from jaxtyping import Array, Float


def hermite_interpolate(x, x0, x1, y0, y1, dy0, dy1):
    """
    Perform Hermite interpolation between points (x0, y0) and (x1, y1)
    with derivatives dy0 and dy1 at those points.
    """
    t = (x - x0) / (x1 - x0)
    dx = x1 - x0
    h00 = 2 * t**3 - 3 * t**2 + 1
    h01 = -2 * t**3 + 3 * t**2
    h10 = t**3 - 2 * t**2 + t
    h11 = t**3 - t**2
    return h00 * y0 + h01 * y1 + h10 * dy0 * dx + h11 * dy1 * dx


class HermiteInterpolation(eqx.Module):
    x0: jax.Array
    x1: jax.Array
    y0: Any
    y1: Any
    dy0: Any
    dy1: Any

    def __call__(self, x):
        """
        Evaluate Hermite spline at t (scalar or array).
        """
        return jax.tree.map(
            lambda y0, y1, dy0, dy1: hermite_interpolate(
                x, self.x0, self.x1, y0, y1, dy0, dy1
            ),
            self.y0,
            self.y1,
            self.dy0,
            self.dy1,
        )


class HermiteSpline(eqx.Module):
    points: jax.Array
    values: Any
    derivatives: Any

    def _find_segment(self, xi):
        idx = jnp.searchsorted(self.points, xi) - 1
        idx = jnp.clip(idx, 0, len(self.points) - 2)
        return idx

    def __call__(self, x: Float[Array, ""]):
        """
        Evaluate piecewise Hermite spline at x (scalar or array).
        """

        idx = self._find_segment(x)
        x0 = self.points[idx]
        x1 = self.points[idx + 1]
        y0 = jax.tree.map(lambda y: y[idx], self.values)
        y1 = jax.tree.map(lambda y: y[idx + 1], self.values)
        dy0 = jax.tree.map(lambda dy: dy[idx], self.derivatives)
        dy1 = jax.tree.map(lambda dy: dy[idx + 1], self.derivatives)
        return hermite_interpolate(x, x0, x1, y0, y1, dy0, dy1)
