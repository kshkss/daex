import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any, NamedTuple
from jaxtyping import Array, Float
import chex


def assert_trees_shape_equal(tree1: Any, tree2: Any):  # , err_msg: str = ""):
    """
    2つのpytreeが構造とテンソルの形状まで完全に一致することを確認

    Args:
        tree1: 最初のpytree
        tree2: 2番目のpytree
        err_msg: エラーメッセージのプレフィックス

    Raises:
        AssertionError: 構造または形状が一致しない場合
    """
    # 構造の一致を確認
    chex.assert_trees_all_equal_structs(tree1, tree2)

    # 各リーフの形状を取得
    leaves1, _ = jax.tree.flatten(tree1)
    leaves2, _ = jax.tree.flatten(tree2)

    # 全てのリーフの形状が一致することを確認
    for leaf1, leaf2 in zip(leaves1, leaves2):
        chex.assert_equal_shape([leaf1, leaf2])


class Quadrature(NamedTuple):
    points: jax.Array
    weights: jax.Array


def divide_intervals(t0: jax.Array, t1: jax.Array, n: int = 4) -> Quadrature:
    t0 = t0.reshape([-1, 1])
    t1 = t1.reshape([-1, 1])
    ws = (
        0.5
        * (t1 - t0)
        * jnp.array(
            [
                0.1666666666666666666667,
                0.8333333333333333333333,
                0.833333333333333333333,
                0.1666666666666666666667,
            ]
        )
    )
    _ts = 0.5 * (t1 + t0) + 0.5 * (t1 - t0) * jnp.array(
        [-1.0, -0.447213595499957939282, 0.447213595499957939282, 1]
    )
    ts = jnp.append(_ts[:, :-1].flatten(), _ts[-1, -1])

    return Quadrature(points=ts, weights=ws)


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
