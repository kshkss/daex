from typing import Callable, Any, NamedTuple
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax.flatten_util import ravel_pytree
from sksundae._cy_ida import IDA as _IDA
import equinox as eqx
from jaxtyping import Array, Float
from daex.utils import HermiteSpline
from daex import utils
from functools import partial


@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _call_ida(
    deriv_fn: Callable,
    const_fn: Callable,
    params: Float[Array, " a_size"],
    ts: Float[Array, " ticks"],
    x0: Float[Array, " x_size"],
    y0: Float[Array, " y_size"],
    **options,
):
    x_size = x0.size
    xy = jnp.concatenate([x0, y0])
    xyp = jnp.concatenate([jnp.zeros_like(x0), deriv_fn(params, ts[0], x0, y0)])
    y_type = jax.ShapeDtypeStruct(list(ts.shape) + list(xy.shape), xy.dtype)
    yp_type = jax.ShapeDtypeStruct(list(ts.shape) + list(xyp.shape), xyp.dtype)

    def residual(params, t, xy, xyp):
        x = xy[:x_size]
        y = xy[x_size:]
        yp = xyp[x_size:]
        res = jnp.concatenate(
            [const_fn(params, t, x, y), yp - deriv_fn(params, t, x, y)]
        )
        return res

    def resfn(t, y, yp, res, userdata):
        (a,) = userdata
        t = jnp.asarray(t)
        y = jnp.asarray(y)
        yp = jnp.asarray(yp)
        res[:] = np.asarray(residual(a, t, y, yp))

    def jacfn(t, y, yp, res, cj, JJ, userdata):
        (a,) = userdata
        t = jnp.asarray(t)
        y = jnp.asarray(y)
        yp = jnp.asarray(yp)
        dy, dyp = jax.jacfwd(residual, argnums=[2, 3], has_aux=False)(a, t, y, yp)
        JJ[:, :] = np.asarray(dy + cj * dyp)

    def _run_forward(
        params: np.ndarray, ts: np.ndarray, y0: np.ndarray, yp0: np.ndarray
    ):
        ida = _IDA(
            resfn,
            jacfn=jacfn,
            userdata=(params,),
            algebraic_idx=np.arange(x_size),
            **options,
        )
        results = ida.solve(ts, y0, yp0)
        if not results.success:
            raise RuntimeError(f"IDA solver failed: {results.message}")
        if ts.shape[0] == 2:
            y = np.take(results.y, np.array([0, -1]), axis=0)
            yp = np.take(results.y, np.array([0, -1]), axis=0)
        else:
            y = results.y
            yp = results.yp
        return y, yp

    xy, xyp = jax.pure_callback(
        _run_forward,
        (y_type, yp_type),
        params,
        ts,
        xy,
        xyp,
        vmap_method="sequential",
    )

    return xy[:, :x_size], xy[:, x_size:], xyp[:, x_size:]


@_call_ida.defjvp
def _call_ida_jvp(deriv_fn, const_fn, primals, tangents) -> tuple:
    params, ts, x0, y0 = primals
    d_params, d_ts, _, dy0 = tangents
    # x, y, yp = _call_ida(deriv_fn, const_fn, params, ts, x0, y0)
    params_array, unravel_a = ravel_pytree((params, d_params))

    yp0 = deriv_fn(params, ts[0], x0, y0)
    z0_a = jnp.zeros_like(y0)
    z0_y0 = dy0
    z0_t0 = -yp0 * d_ts[0]

    def deriv_ext(
        params_array: jax.Array,
        t: jax.Array,
        x: jax.Array,
        y: jax.Array,
    ):
        params, d_params = unravel_a(params_array)
        y, z_a, z_y0, z_t0 = y.reshape([4, -1])
        yp = deriv_fn(params, t, x, y)

        dfda, dfdx, dfdy = jax.jacrev(deriv_fn, argnums=[0, 2, 3])(params, t, x, y)
        dgda, dgdx, dgdy = jax.jacrev(const_fn, argnums=[0, 2, 3])(params, t, x, y)
        lu_dgdx = jsp.linalg.lu_factor(dgdx)

        dxdz_a = jsp.linalg.lu_solve(lu_dgdx, dgdy @ z_a)
        dxdz_y0 = jsp.linalg.lu_solve(lu_dgdx, dgdy @ z_y0)
        dxdz_t0 = jsp.linalg.lu_solve(lu_dgdx, dgdy @ z_t0)
        zp_a1 = dfdy @ z_a - dfdx @ dxdz_a
        zp_y0 = dfdy @ z_y0 - dfdx @ dxdz_y0
        zp_t0 = dfdy @ z_t0 - dfdx @ dxdz_t0

        dxda = jsp.linalg.lu_solve(lu_dgdx, dgda @ d_params)
        zp_a2 = dfda @ d_params - dfdx @ dxda

        return jnp.concatenate([yp, zp_a1 + zp_a2, zp_y0, zp_t0])

    def const_ext(
        params_array: jax.Array,
        t: jax.Array,
        x: jax.Array,
        y: jax.Array,
    ):
        params, _ = unravel_a(params_array)
        y, _, _, _ = y.reshape([4, -1])
        g = const_fn(params, t, x, y)
        return g

    x, y, yp = _call_ida(
        deriv_ext,
        const_ext,
        params_array,
        ts,
        x0,
        jnp.concatenate([y0, z0_a, z0_y0, z0_t0]),
    )
    y, z_a, z_y0, z_t0 = jnp.transpose(y.reshape([ts.shape[0], 4, -1]), (1, 0, 2))
    yp, zp_a, zp_y0, zp_t0 = jnp.transpose(yp.reshape([ts.shape[0], 4, -1]), (1, 0, 2))
    dx, dy, dyp = jax.vmap(
        _call_ida_jvp_foreach_step,
        in_axes=(None, None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )(
        deriv_fn,
        const_fn,
        params,
        d_params,
        ts,
        d_ts,
        x,
        y,
        z_a,
        z_y0,
        z_t0,
        yp,
        zp_a,
        zp_y0,
        zp_t0,
    )

    return (x, y, yp), (dx, dy, dyp)


def _call_ida_jvp_foreach_step(
    deriv_fn,
    const_fn,
    params,
    d_params,
    t,
    dt,
    x,
    y,
    z_a,
    z_y0,
    z_t0,
    yp,
    zp_a,
    zp_y0,
    zp_t0,
):
    dfdt, dfdx, dfdy = jax.jacrev(deriv_fn, argnums=[1, 2, 3])(params, t, x, y)
    dgda, dgdt, dgdx, dgdy = jax.jacrev(const_fn, argnums=[0, 1, 2, 3])(params, t, x, y)
    lu_dgdx = jsp.linalg.lu_factor(dgdx)

    dy = z_a + z_y0 + z_t0 + yp * t
    dyp = (
        zp_a
        + zp_y0
        + zp_t0
        + (dfdy @ yp + dfdt - dfdx @ jsp.linalg.lu_solve(lu_dgdx, dgdt + dgdy @ yp))
        * dt
    )
    dx = -jsp.linalg.lu_solve(
        lu_dgdx,
        dgdy @ (z_a + z_y0 + z_t0) + dgda @ d_params + (dgdy @ yp + dgdt) * dt,
    )
    return (dx, dy, dyp)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _daeint(
    deriv_fn: Callable,
    const_fn: Callable,
    params: Float[Array, " a_size"],
    ts: Float[Array, " points"],
    x0: Float[Array, " x_size"],
    y0: Float[Array, " y_size"],
) -> tuple[
    Float[Array, " x_size"],
    Float[Array, " y_size"],
    Float[Array, " y_size"],
]:
    """Perform the first step of DAE integration using IDA solver."""

    return _call_ida(deriv_fn, const_fn, params, ts, x0, y0)


def _daeint_fwd(
    deriv_fn: Callable,
    const_fn: Callable,
    params: Float[Array, " a_size"],
    ts: Float[Array, " points"],
    x0: Float[Array, " x_size"],
    y0: Float[Array, " y_size"],
) -> tuple[
    tuple[
        Float[Array, "points x_size"],
        Float[Array, "points y_size"],
        Float[Array, "points y_size"],
    ],
    tuple[
        Float[Array, " a_size"],
        Float[Array, " interpolated"],
        Float[Array, "n_intervals quad_order"],
        Float[Array, "interpolated x_size"],
        Float[Array, "interpolated y_size"],
        Float[Array, "interpolated y_size"],
    ],
]:
    ts, ws = utils.divide_intervals(ts[:-1], ts[1:], n=4)
    x, y, yp = _call_ida(deriv_fn, const_fn, params, ts, x0, y0)
    x1 = x[::3]
    y1 = y[::3]
    yp1 = yp[::3]
    return (x1, y1, yp1), (params, ts, ws, x, y, yp)


def _daeint_bwd(
    deriv_fn: Callable,
    const_fn: Callable,
    residuals: tuple[
        Float[Array, " a_size"],
        Float[Array, " interpolated"],
        Float[Array, "n_intervals quad_order"],
        Float[Array, "interpolated x_size"],
        Float[Array, "interpolated y_size"],
        Float[Array, "interpolated y_size"],
    ],
    cotangents: tuple[
        Float[Array, "points x_size"],
        Float[Array, "points y_size"],
        Float[Array, "points y_size"],
    ],
) -> tuple[
    Float[Array, " a_size"],  # parmas
    Float[Array, " points"],  # ts
    None,  # x0
    Float[Array, " y_size"],  # y0
]:
    params, ts, ws, x, y, yp = residuals
    wx, wy, wyp = cotangents
    ts = ts[::-1]
    ws = ws[::-1]
    x = x[::-1]
    y = y[::-1]
    yp = yp[::-1]
    wx = wx[::-1]
    wy = wy[::-1]
    wyp = wyp[::-1]
    n = 4
    points = ts[::3].shape[0]

    def adjfn(
        params_array: jax.Array,
        t: jax.Array,
        xarray: jax.Array,
        yarray: jax.Array,
        zarray: jax.Array,
    ):
        dfdx, dfdy = jax.jacfwd(deriv_fn, argnums=[2, 3])(
            params_array, t, xarray, yarray
        )
        dgdx, dgdy = jax.jacfwd(const_fn, argnums=[2, 3])(
            params_array, t, xarray, yarray
        )
        dfdz = dfdy - dfdx @ jnp.linalg.solve(dgdx, dgdy)
        dz = -dfdz.T @ zarray
        return dz

    def da_fn(
        params_array: jax.Array,
        t: jax.Array,
        xarray: jax.Array,
        yarray: jax.Array,
        zarray: jax.Array,
    ) -> jax.Array:
        dfda, dfdx = jax.jacrev(deriv_fn, argnums=[0, 2])(
            params_array, t, xarray, yarray
        )
        dgda, dgdx = jax.jacrev(const_fn, argnums=[0, 2])(
            params_array, t, xarray, yarray
        )
        lu_dgdx = jsp.linalg.lu_factor(dgdx)
        dfda_x = dfdx @ jsp.linalg.lu_solve(lu_dgdx, dgda)
        da = (dfda - dfda_x).T @ zarray
        return da

    def body(i, carry):
        dJda, dJdt0_prev, dJdt, dJdy0 = carry
        dJda0, dJdt1, dJdt0, _, dJdy0 = _daeint_bwd_step(
            deriv_fn,
            const_fn,
            adjfn,
            da_fn,
            residuals=(
                params,
                jax.lax.dynamic_slice_in_dim(ts, i * (n - 1), n)[::-1],
                ws[i],
                jax.lax.dynamic_slice_in_dim(x, i * (n - 1), n)[::-1],
                jax.lax.dynamic_slice_in_dim(y, i * (n - 1), n)[::-1],
                jax.lax.dynamic_slice_in_dim(yp, i * (n - 1), n)[::-1],
            ),
            cotangents=(
                wx[i],
                wy[i] + dJdy0,
                wyp[i],
            ),
        )
        dJdt = dJdt.at[i].set(dJdt1 + dJdt0_prev)
        dJdt0_prev = dJdt0
        dJda = dJda + dJda0
        return dJda, dJdt0_prev, dJdt, dJdy0

    dJda, dJdt0_prev, dJdt, dJdy0 = jax.lax.fori_loop(
        0,
        points - 1,
        body,
        (
            jnp.zeros_like(params),
            0.0,
            jnp.zeros(points),
            jnp.zeros_like(wy[0]),
        ),
    )
    dJdt = dJdt.at[-1].set(dJdt0_prev)

    return (dJda, dJdt[::-1], None, dJdy0)


def _daeint_bwd_step(
    deriv_fn: Callable,
    const_fn: Callable,
    adjfn: Callable,
    da_fn: Callable,
    residuals: tuple[
        Float[Array, " a_size"],
        Float[Array, " quad_order"],
        Float[Array, " quad_order"],
        Float[Array, "quad_order x_size"],
        Float[Array, "quad_order y_size"],
        Float[Array, "quad_order y_size"],
    ],
    cotangents: tuple[
        Float[Array, " x_size"], Float[Array, " y_size"], Float[Array, " y_size"]
    ],
) -> tuple[
    Float[Array, " a_size"],  # parmas
    Float[Array, ""],  # t1
    Float[Array, ""],  # t0
    None,  # x0
    Float[Array, " y_size"],  # y0
]:
    params, ts, ws, x, y, yp = residuals
    wx, wy, wyp = cotangents
    t1 = ts[-1]
    x1 = x[-1]
    y1 = y[-1]
    yp1 = yp[-1]
    yfunc = HermiteSpline(ts, y, yp)
    params_array, unravel_a = ravel_pytree((params, yfunc))

    dgda, dgdt, dgdx, dgdy = jax.jacrev(const_fn, argnums=[0, 1, 2, 3])(
        params, t1, x1, y1
    )
    dfda, dfdt, dfdx, dfdy = jax.jacrev(deriv_fn, argnums=[0, 1, 2, 3])(
        params, t1, x1, y1
    )
    lu_dgdx = jsp.linalg.lu_factor(dgdx)
    wxx = wx + dfdx.T @ wyp
    dJdt = (
        jnp.dot(wy, yp1)
        + jnp.dot(wyp, dfdt + dfdy @ yp1)
        - jnp.dot(wxx, jsp.linalg.lu_solve(lu_dgdx, dgdt + dgdy @ yp1))
    )
    dJda = jnp.dot(wyp, dfda) - jnp.dot(wxx, jsp.linalg.lu_solve(lu_dgdx, dgda))
    z1 = wy + jnp.dot(wyp, dfdy) - jnp.dot(wxx, jsp.linalg.lu_solve(lu_dgdx, dgdy))

    def deriv_adj(
        params_array: jax.Array,
        t: jax.Array,
        x: jax.Array,
        z: jax.Array,
    ):
        params, yfunc = unravel_a(params_array)
        y = yfunc(t)
        # dz = dfdy - dfdx @ jsp.linalg.lu_solve(lu_dgdx, dgdy)
        # zp = -dz.T @ z
        zp = adjfn(params, t, x, y, z)
        return zp

    def const_adj(
        params_array: jax.Array,
        t: jax.Array,
        x: jax.Array,
        z: jax.Array,
    ):
        params, yfunc = unravel_a(params_array)
        y = yfunc(t)
        return const_fn(params, t, x, y)

    _, z, _ = _call_ida(deriv_adj, const_adj, params_array, ts[::-1], x1, z1)
    z = z[::-1]
    integral = jax.vmap(da_fn, in_axes=(None, 0, 0, 0, 0))(params, ts, x, y, z)
    dJda = dJda + jnp.dot(ws, integral)

    return (dJda, dJdt, -jnp.dot(z[0], yp[0]), None, z[0])


_daeint.defvjp(_daeint_fwd, _daeint_bwd)


class Results[U](NamedTuple):
    values: U
    derivatives: U


class SemiExplicitDAE(eqx.Module):
    partition: Callable
    deriv_fn: Callable
    const_fn: Callable
    run_forward: Callable
    deriv_ext: Callable
    run_forward_extend: Callable
    deriv_adj: Callable
    const_adj: Callable
    run_adjoint: Callable
    deriv_adj_ext: Callable
    run_adjoint_extend: Callable
    da_fn: Callable
    _clear_cache: Callable

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._clear_cache()

    def __del__(self):
        self._clear_cache()


def def_semi_explicit_dae[Params, Var](
    derivative: Callable[[Params, jax.Array, Var], Var],
    constraint: Callable[[Params, jax.Array, Var], Any],
    params: Params,
    t0: jax.Array,
    y0: Var,
):
    """
    Function to define a system by
    explicit ODE like as y' = f(t, y),
    or semi-explicit DAE like as y' = f(t, x, y), g(t, x, y) = 0.

    Args:
    - deriv_fn (Callable): A function that takes parameters, a coordinate `t`, and variables `x` and `y`.
      It returns the derivative `y'` of the differential variables `y`. The parameters and variables are pytrees.
      The return value `y'` is a pytree with the same structure as the input `x` and `y`,
      but with `None` in the positions corresponding to the algebraic variables `x`.

    - const_fn (Callable): A function that takes the same arguments as `deriv_fn` and returns the residuals
      of the constraints for the algebraic variables. The algebraic variables are computed such that the return value
      of `const_fn` becomes zero. If you want to solve an explicit ODE, you can pass a function that returns `None`.

    - params (Params): Parameters used in `deriv_fn` and `const_fn`. It can be a pytree.

    - t0 (jax.Array): Initial coordinate.

    - y0 (Var): Initial values of the variables. It is a pytree containing both differential and algebraic variables.
    """
    yp0 = derivative(params, t0, y0)
    is_algebraic = jax.tree.map(lambda _, yp: yp is None, y0, yp0)

    def partition(xy: Var) -> tuple[Var, Var]:
        x, y = eqx.partition(xy, is_algebraic, is_leaf=eqx.is_inexact_array)
        return x, y

    x0, y0 = partition(y0)
    try:
        utils.assert_trees_shape_equal(y0, yp0)
    except AssertionError as e:
        raise ValueError(
            "The shapes of initial conditions of differential variables and their derivative do not match. "
            "Check the initial conditions and deriv_fn()."
        ) from e
    x, unravel_x = ravel_pytree(x0)
    _, unravel_y = ravel_pytree(y0)
    _, unravel_a = ravel_pytree(params)
    x_size = x.size

    def deriv_fn(
        params_array: jax.Array, t: jax.Array, xarray: jax.Array, yarray: jax.Array
    ) -> jax.Array:
        params = unravel_a(params_array)
        x = unravel_x(xarray)
        y = unravel_y(yarray)
        xy = eqx.combine(x, y)
        yp = derivative(params, t, xy)
        yparray, _ = ravel_pytree(yp)
        return yparray

    def const_fn(
        params_array: jax.Array, t: jax.Array, xarray: jax.Array, yarray: jax.Array
    ) -> jax.Array:
        params = unravel_a(params_array)
        x = unravel_x(xarray)
        y = unravel_y(yarray)
        xy = eqx.combine(x, y)
        g = constraint(params, t, xy)
        garray, _ = ravel_pytree(g)
        return garray

    @jax.jit
    def residual(params, t, xy, xyp):
        x = xy[:x_size]
        y = xy[x_size:]
        yp = xyp[x_size:]
        res = jnp.concatenate(
            [const_fn(params, t, x, y), yp - deriv_fn(params, t, x, y)]
        )
        return res

    def resfn(t, y, yp, res, userdata):
        (a,) = userdata
        t = jnp.asarray(t)
        y = jnp.asarray(y)
        yp = jnp.asarray(yp)
        res[:] = np.asarray(residual(a, t, y, yp))

    def jacfn(t, y, yp, res, cj, JJ, userdata):
        (a,) = userdata
        t = jnp.asarray(t)
        y = jnp.asarray(y)
        yp = jnp.asarray(yp)
        dy, dyp = jax.jacfwd(residual, argnums=[2, 3], has_aux=False)(a, t, y, yp)
        JJ[:, :] = np.asarray(dy + cj * dyp)

    def run_forward(
        params: np.ndarray, ts: np.ndarray, y0: np.ndarray, yp0: np.ndarray
    ):
        ida = _IDA(
            resfn,
            jacfn=jacfn,
            userdata=(params,),
            algebraic_idx=np.arange(x_size),
        )
        results = ida.solve(ts, y0, yp0)
        if not results.success:
            raise RuntimeError(f"IDA solver failed: {results.message}")
        if ts.shape[0] == 2:
            y = np.take(results.y, np.array([0, -1]), axis=0)
            yp = np.take(results.y, np.array([0, -1]), axis=0)
        else:
            y = results.y
            yp = results.yp
        return y, yp

    def deriv_ext(
        params_array: tuple[jax.Array, jax.Array],
        t: jax.Array,
        x: jax.Array,
        y: jax.Array,
    ):
        params, d_params = params_array
        y, z_a, z_y0, z_t0 = y.reshape([4, -1])
        yp = deriv_fn(params, t, x, y)

        dfda, dfdx, dfdy = jax.jacrev(deriv_fn, argnums=[0, 2, 3])(params, t, x, y)
        dgda, dgdx, dgdy = jax.jacrev(const_fn, argnums=[0, 2, 3])(params, t, x, y)
        lu_dgdx = jsp.linalg.lu_factor(dgdx)

        dxdz_a = jsp.linalg.lu_solve(lu_dgdx, dgdy @ z_a)
        dxdz_y0 = jsp.linalg.lu_solve(lu_dgdx, dgdy @ z_y0)
        dxdz_t0 = jsp.linalg.lu_solve(lu_dgdx, dgdy @ z_t0)
        zp_a1 = dfdy @ z_a - dfdx @ dxdz_a
        zp_y0 = dfdy @ z_y0 - dfdx @ dxdz_y0
        zp_t0 = dfdy @ z_t0 - dfdx @ dxdz_t0

        dxda = jsp.linalg.lu_solve(lu_dgdx, dgda @ d_params)
        zp_a2 = dfda @ d_params - dfdx @ dxda

        return jnp.concatenate([yp, zp_a1 + zp_a2, zp_y0, zp_t0])

    def const_ext(
        params_array: tuple[jax.Array, jax.Array],
        t: jax.Array,
        x: jax.Array,
        y: jax.Array,
    ):
        params, _ = params_array
        y, _, _, _ = y.reshape([4, -1])
        g = const_fn(params, t, x, y)
        return g

    @jax.jit
    def residual_ext(params, t, xy, xyp):
        x = xy[:x_size]
        y = xy[x_size:]
        yp = xyp[x_size:]
        res = jnp.concatenate(
            [const_ext(params, t, x, y), yp - deriv_ext(params, t, x, y)]
        )
        return res

    def resfn_ext(t, y, yp, res, userdata):
        a = jnp.asarray(userdata[0])
        da = jnp.asarray(userdata[1])
        t = jnp.asarray(t)
        y = jnp.asarray(y)
        yp = jnp.asarray(yp)
        res[:] = np.asarray(residual_ext((a, da), t, y, yp))

    def jacfn_ext(t, y, yp, res, cj, JJ, userdata):
        a = jnp.asarray(userdata[0])
        da = jnp.asarray(userdata[1])
        t = jnp.asarray(t)
        y = jnp.asarray(y)
        yp = jnp.asarray(yp)
        dy, dyp = jax.jacfwd(residual_ext, argnums=[2, 3], has_aux=False)(
            (a, da), t, y, yp
        )
        JJ[:, :] = np.asarray(dy + cj * dyp)

    def run_forward_extend(
        params: tuple[np.ndarray, np.ndarray],
        ts: np.ndarray,
        y0: np.ndarray,
        yp0: np.ndarray,
    ):
        ida = _IDA(
            resfn_ext,
            jacfn=jacfn_ext,
            userdata=params,
            algebraic_idx=np.arange(x_size),
        )
        results = ida.solve(ts, y0, yp0)
        if not results.success:
            raise RuntimeError(f"IDA solver failed: {results.message}")
        if ts.shape[0] == 2:
            y = np.take(results.y, np.array([0, -1]), axis=0)
            yp = np.take(results.y, np.array([0, -1]), axis=0)
        else:
            y = results.y
            yp = results.yp
        return y, yp

    def adjfn(
        params_array: jax.Array,
        t: jax.Array,
        xarray: jax.Array,
        yarray: jax.Array,
        zarray: jax.Array,
    ):
        dfdx, dfdy = jax.jacfwd(deriv_fn, argnums=[2, 3])(
            params_array, t, xarray, yarray
        )
        dgdx, dgdy = jax.jacfwd(const_fn, argnums=[2, 3])(
            params_array, t, xarray, yarray
        )
        dfdz = dfdy - dfdx @ jnp.linalg.solve(dgdx, dgdy)
        dz = -dfdz.T @ zarray
        return dz

    def da_fn(
        params_array: jax.Array,
        t: jax.Array,
        xarray: jax.Array,
        yarray: jax.Array,
        zarray: jax.Array,
    ) -> jax.Array:
        dfda, dfdx = jax.jacrev(deriv_fn, argnums=[0, 2])(
            params_array, t, xarray, yarray
        )
        dgda, dgdx = jax.jacrev(const_fn, argnums=[0, 2])(
            params_array, t, xarray, yarray
        )
        lu_dgdx = jsp.linalg.lu_factor(dgdx)
        dfda_x = dfdx @ jsp.linalg.lu_solve(lu_dgdx, dgda)
        da = (dfda - dfda_x).T @ zarray
        return da

    def deriv_adj(
        params: jax.Array,
        t: jax.Array,
        x: jax.Array,
        z: jax.Array,
        yfunc: HermiteSpline,
    ):
        y = yfunc(t)
        zp = adjfn(params, t, x, y, z)
        return zp

    def const_adj(
        params: jax.Array,
        t: jax.Array,
        x: jax.Array,
        z: jax.Array,
        yfunc: HermiteSpline,
    ):
        y = yfunc(t)
        return const_fn(params, t, x, y)

    @jax.jit
    def residual_adj(params_set, t, xy, xyp):
        params, yfunc = params_set
        x = xy[:x_size]
        z = xy[x_size:]
        zp = xyp[x_size:]
        res = jnp.concatenate(
            [const_adj(params, t, x, z, yfunc), zp - deriv_adj(params, t, x, z, yfunc)]
        )
        return res

    def resfn_adj(t, y, yp, res, userdata):
        t = jnp.asarray(t)
        y = jnp.asarray(y)
        yp = jnp.asarray(yp)
        res[:] = np.asarray(residual_adj(userdata, t, y, yp))

    def jacfn_adj(t, y, yp, res, cj, JJ, userdata):
        t = jnp.asarray(t)
        y = jnp.asarray(y)
        yp = jnp.asarray(yp)
        dy, dyp = jax.jacfwd(residual_adj, argnums=[2, 3], has_aux=False)(
            userdata, t, y, yp
        )
        JJ[:, :] = np.asarray(dy + cj * dyp)

    def run_adjoint(
        params: tuple[np.ndarray, HermiteSpline],
        ts: np.ndarray,
        y0: np.ndarray,
        yp0: np.ndarray,
    ):
        ida = _IDA(
            resfn_adj,
            jacfn=jacfn_adj,
            userdata=params,
            algebraic_idx=np.arange(x_size),
            calc_initcond="yp0",
            calc_init_dt=-0.01,
        )
        results = ida.solve(ts, y0, yp0)
        if not results.success:
            raise RuntimeError(f"IDA solver failed: {results.message}")
        if ts.shape[0] == 2:
            y = np.take(results.y, np.array([0, -1]), axis=0)
            yp = np.take(results.y, np.array([0, -1]), axis=0)
        else:
            y = results.y
            yp = results.yp
        return y, yp

    def deriv_adj_ext(
        params_array: tuple[jax.Array, jax.Array, HermiteSpline],
        t: jax.Array,
        x: jax.Array,
        z: jax.Array,
    ):
        params, d_params, yfunc = params_array
        z, z_a, z_y0, z_t0 = z.reshape([4, -1])
        yp = deriv_adj(params, t, x, z, yfunc)

        dfda, dfdx, dfdy = jax.jacrev(deriv_adj, argnums=[0, 2, 3])(
            params, t, x, z, yfunc
        )
        dgda, dgdx, dgdy = jax.jacrev(const_adj, argnums=[0, 2, 3])(
            params, t, x, z, yfunc
        )
        lu_dgdx = jsp.linalg.lu_factor(dgdx)

        dxdz_a = jsp.linalg.lu_solve(lu_dgdx, dgdy @ z_a)
        dxdz_y0 = jsp.linalg.lu_solve(lu_dgdx, dgdy @ z_y0)
        dxdz_t0 = jsp.linalg.lu_solve(lu_dgdx, dgdy @ z_t0)
        zp_a1 = dfdy @ z_a - dfdx @ dxdz_a
        zp_y0 = dfdy @ z_y0 - dfdx @ dxdz_y0
        zp_t0 = dfdy @ z_t0 - dfdx @ dxdz_t0

        dxda = jsp.linalg.lu_solve(lu_dgdx, dgda @ d_params)
        zp_a2 = dfda @ d_params - dfdx @ dxda

        return jnp.concatenate([yp, zp_a1 + zp_a2, zp_y0, zp_t0])

    def const_adj_ext(
        params_array: tuple[jax.Array, jax.Array, HermiteSpline],
        t: jax.Array,
        x: jax.Array,
        y: jax.Array,
    ):
        params, _, yfunc = params_array
        y, _, _, _ = y.reshape([4, -1])
        g = const_adj(params, t, x, y, yfunc)
        return g

    @jax.jit
    def residual_adj_ext(params, t, xy, xyp):
        x = xy[:x_size]
        y = xy[x_size:]
        yp = xyp[x_size:]
        res = jnp.concatenate(
            [
                const_adj_ext(params, t, x, y),
                yp - deriv_adj_ext(params, t, x, y),
            ]
        )
        return res

    def resfn_adj_ext(t, y, yp, res, userdata):
        t = jnp.asarray(t)
        y = jnp.asarray(y)
        yp = jnp.asarray(yp)
        res[:] = np.asarray(residual_adj_ext(userdata, t, y, yp))

    def jacfn_adj_ext(t, y, yp, res, cj, JJ, userdata):
        t = jnp.asarray(t)
        y = jnp.asarray(y)
        yp = jnp.asarray(yp)
        dy, dyp = jax.jacfwd(residual_adj_ext, argnums=[2, 3], has_aux=False)(
            userdata, t, y, yp
        )
        JJ[:, :] = np.asarray(dy + cj * dyp)

    def run_adjoint_extend(
        params: tuple[np.ndarray, np.ndarray, HermiteSpline],
        ts: np.ndarray,
        y0: np.ndarray,
        yp0: np.ndarray,
    ):
        ida = _IDA(
            resfn_adj_ext,
            jacfn=jacfn_adj_ext,
            userdata=params,
            algebraic_idx=np.arange(x_size),
        )
        results = ida.solve(ts, y0, yp0)
        if not results.success:
            raise RuntimeError(f"IDA solver failed: {results.message}")
        if ts.shape[0] == 2:
            y = np.take(results.y, np.array([0, -1]), axis=0)
            yp = np.take(results.y, np.array([0, -1]), axis=0)
        else:
            y = results.y
            yp = results.yp
        return y, yp

    def clear_cache():
        residual._clear_cache()
        residual_ext._clear_cache()
        residual_adj._clear_cache()
        residual_adj_ext._clear_cache()

    return SemiExplicitDAE(
        partition=partition,
        deriv_fn=deriv_fn,
        const_fn=const_fn,
        run_forward=run_forward,
        deriv_ext=deriv_ext,
        run_forward_extend=run_forward_extend,
        deriv_adj=deriv_adj,
        const_adj=const_adj,
        run_adjoint=run_adjoint,
        deriv_adj_ext=deriv_adj_ext,
        run_adjoint_extend=run_adjoint_extend,
        da_fn=da_fn,
        _clear_cache=clear_cache,
    )


def _finalize_jvp(deriv_fn, const_fn, params, d_params, t, dt, x, y, yp, *args):
    y, z_a, z_y0, z_t0 = y
    yp, zp_a, zp_y0, zp_t0 = yp
    dfdt, dfdx, dfdy = jax.jacrev(deriv_fn, argnums=[1, 2, 3])(params, t, x, y, *args)
    dgda, dgdt, dgdx, dgdy = jax.jacrev(const_fn, argnums=[0, 1, 2, 3])(
        params, t, x, y, *args
    )
    lu_dgdx = jsp.linalg.lu_factor(dgdx)

    dy = z_a + z_y0 + z_t0 + yp * t
    dyp = (
        zp_a
        + zp_y0
        + zp_t0
        + (dfdy @ yp + dfdt - dfdx @ jsp.linalg.lu_solve(lu_dgdx, dgdt + dgdy @ yp))
        * dt
    )
    dx = -jsp.linalg.lu_solve(
        lu_dgdx,
        dgdy @ (z_a + z_y0 + z_t0) + dgda @ d_params + (dgdy @ yp + dgdt) * dt,
    )
    return (dx, dy, dyp)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _daeint2(
    callbacks: SemiExplicitDAE,
    params: Float[Array, " a_size"],
    ts: Float[Array, " points"],
    x0: Float[Array, " x_size"],
    y0: Float[Array, " y_size"],
) -> tuple[
    Float[Array, " x_size"],
    Float[Array, " y_size"],
    Float[Array, " y_size"],
]:
    """Perform DAE integration using IDA."""

    x, y, yp = run_forward(callbacks, params, ts, x0, y0)
    return x, y, yp


def _daeint_fwd2(
    callbacks: SemiExplicitDAE,
    params: Float[Array, " a_size"],
    ts: Float[Array, " points"],
    x0: Float[Array, " x_size"],
    y0: Float[Array, " y_size"],
) -> tuple[
    tuple[
        Float[Array, "points x_size"],
        Float[Array, "points y_size"],
        Float[Array, "points y_size"],
    ],
    tuple[
        Float[Array, " a_size"],
        Float[Array, " interpolated"],
        Float[Array, "n_intervals quad_order"],
        Float[Array, "interpolated x_size"],
        Float[Array, "interpolated y_size"],
        Float[Array, "interpolated y_size"],
    ],
]:
    ts, ws = utils.divide_intervals(ts[:-1], ts[1:], n=4)
    x, y, yp = run_forward(callbacks, params, ts, x0, y0)
    x1 = x[::3]
    y1 = y[::3]
    yp1 = yp[::3]
    return (x1, y1, yp1), (params, ts, ws, x, y, yp)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def run_forward(callbacks: SemiExplicitDAE, params, ts, x0, y0):
    yp0 = callbacks.deriv_fn(params, ts[0], x0, y0)
    xy = jnp.append(x0, y0)
    xyp = jnp.append(jnp.zeros_like(x0), yp0)
    y_type = jax.ShapeDtypeStruct(list(ts.shape) + list(xy.shape), xy.dtype)
    yp_type = jax.ShapeDtypeStruct(list(ts.shape) + list(xyp.shape), xyp.dtype)

    xy, xyp = jax.pure_callback(
        callbacks.run_forward,
        (y_type, yp_type),
        params,
        ts,
        xy,
        xyp,
        vmap_method="sequential",
    )

    x = xy[:, : x0.size]
    y = xy[:, x0.size :]
    yp = xyp[:, x0.size :]

    return x, y, yp


@run_forward.defjvp
def run_forward_jvp(callbacks: SemiExplicitDAE, primals, tangents):
    params, ts, x0, y0 = primals
    d_params, d_ts, _, d_y0 = tangents
    yp0 = callbacks.deriv_fn(params, ts[0], x0, y0)
    z_a = jnp.zeros_like(y0)
    z_y0 = d_y0
    z_t0 = -yp0 * d_ts[0]

    z0 = jnp.concatenate([y0, z_a, z_y0, z_t0])
    xy = jnp.append(x0, z0)
    xyp = jnp.append(
        jnp.zeros_like(x0), callbacks.deriv_ext((params, d_params), ts[0], x0, z0)
    )

    y_type = jax.ShapeDtypeStruct(list(ts.shape) + list(xy.shape), xy.dtype)
    yp_type = jax.ShapeDtypeStruct(list(ts.shape) + list(xyp.shape), xyp.dtype)

    xy, xyp = jax.pure_callback(
        callbacks.run_forward_extend,
        (y_type, yp_type),
        (params, d_params),
        ts,
        xy,
        xyp,
        vmap_method="sequential",
    )
    x = xy[:, : x0.size]
    y = xy[:, x0.size :].reshape([ts.size, 4, y0.size])
    yp = xyp[:, x0.size :].reshape([ts.size, 4, y0.size])

    dx, dy, dyp = jax.vmap(
        _finalize_jvp, in_axes=(None, None, None, None, 0, 0, 0, 0, 0)
    )(
        callbacks.deriv_fn,
        callbacks.const_fn,
        params,
        d_params,
        ts,
        d_ts,
        x,
        y,
        yp,
    )
    return (x, y[:, 0, :], yp[:, 0, :]), (dx, dy, dyp)


def _daeint_bwd2(
    callbacks: SemiExplicitDAE,
    residuals: tuple[
        Float[Array, " a_size"],
        Float[Array, " interpolated"],
        Float[Array, "n_intervals quad_order"],
        Float[Array, "interpolated x_size"],
        Float[Array, "interpolated y_size"],
        Float[Array, "interpolated y_size"],
    ],
    cotangents: tuple[
        Float[Array, "points x_size"],
        Float[Array, "points y_size"],
        Float[Array, "points y_size"],
    ],
) -> tuple[
    Float[Array, " a_size"],  # parmas
    Float[Array, " points"],  # ts
    None,  # x0
    Float[Array, " y_size"],  # y0
]:
    params, ts, ws, x, y, yp = residuals
    wx, wy, wyp = cotangents
    ts = ts[::-1]
    ws = ws[::-1]
    x = x[::-1]
    y = y[::-1]
    yp = yp[::-1]
    wx = wx[::-1]
    wy = wy[::-1]
    wyp = wyp[::-1]
    n = 4
    points = ts[::3].shape[0]

    def body(i, carry):
        dJda, dJdt0_prev, dJdt, dJdy0 = carry
        dJda0, dJdt1, dJdt0, _, dJdy0 = _daeint_bwd_step2(
            callbacks,
            residuals=(
                params,
                jax.lax.dynamic_slice_in_dim(ts, i * (n - 1), n)[::-1],
                ws[i],
                jax.lax.dynamic_slice_in_dim(x, i * (n - 1), n)[::-1],
                jax.lax.dynamic_slice_in_dim(y, i * (n - 1), n)[::-1],
                jax.lax.dynamic_slice_in_dim(yp, i * (n - 1), n)[::-1],
            ),
            cotangents=(
                wx[i],
                wy[i] + dJdy0,
                wyp[i],
            ),
        )
        dJdt = dJdt.at[i].set(dJdt1 + dJdt0_prev)
        dJdt0_prev = dJdt0
        dJda = dJda + dJda0
        return dJda, dJdt0_prev, dJdt, dJdy0

    dJda, dJdt0_prev, dJdt, dJdy0 = jax.lax.fori_loop(
        0,
        points - 1,
        body,
        (
            jnp.zeros_like(params),
            0.0,
            jnp.zeros(points),
            jnp.zeros_like(wy[0]),
        ),
    )
    dJdt = dJdt.at[-1].set(dJdt0_prev)

    return (dJda, dJdt[::-1], None, dJdy0)


_daeint2.defvjp(_daeint_fwd2, _daeint_bwd2)


def _daeint_bwd_step2(
    callbacks: SemiExplicitDAE,
    residuals: tuple[
        Float[Array, " a_size"],
        Float[Array, " quad_order"],
        Float[Array, " quad_order"],
        Float[Array, "quad_order x_size"],
        Float[Array, "quad_order y_size"],
        Float[Array, "quad_order y_size"],
    ],
    cotangents: tuple[
        Float[Array, " x_size"], Float[Array, " y_size"], Float[Array, " y_size"]
    ],
) -> tuple[
    Float[Array, " a_size"],  # parmas
    Float[Array, ""],  # t1
    Float[Array, ""],  # t0
    None,  # x0
    Float[Array, " y_size"],  # y0
]:
    params, ts, ws, x, y, yp = residuals
    wx, wy, wyp = cotangents
    t1 = ts[-1]
    x1 = x[-1]
    y1 = y[-1]
    yp1 = yp[-1]
    yfunc = HermiteSpline(ts, y, yp)

    dgda, dgdt, dgdx, dgdy = jax.jacrev(callbacks.const_fn, argnums=[0, 1, 2, 3])(
        params, t1, x1, y1
    )
    dfda, dfdt, dfdx, dfdy = jax.jacrev(callbacks.deriv_fn, argnums=[0, 1, 2, 3])(
        params, t1, x1, y1
    )
    lu_dgdx = jsp.linalg.lu_factor(dgdx)
    wxx = wx + dfdx.T @ wyp
    dJdt = (
        jnp.dot(wy, yp1)
        + jnp.dot(wyp, dfdt + dfdy @ yp1)
        - jnp.dot(wxx, jsp.linalg.lu_solve(lu_dgdx, dgdt + dgdy @ yp1))
    )
    dJda = jnp.dot(wyp, dfda) - jnp.dot(wxx, jsp.linalg.lu_solve(lu_dgdx, dgda))
    z1 = wy + jnp.dot(wyp, dfdy) - jnp.dot(wxx, jsp.linalg.lu_solve(lu_dgdx, dgdy))

    z = run_adjoint(callbacks, yfunc, params, ts, x1, z1)

    integral = jax.vmap(callbacks.da_fn, in_axes=(None, 0, 0, 0, 0))(
        params, ts, x, y, z
    )
    dJda = dJda + jnp.dot(ws, integral)

    return (dJda, dJdt, -jnp.dot(z[0], yp[0]), None, z[0])


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def run_adjoint(callbacks: SemiExplicitDAE, yfunc, params, ts, x1, z1):
    zp1 = callbacks.deriv_adj(params, ts[-1], x1, z1, yfunc)
    xz = jnp.append(x1, z1)
    xzp = jnp.append(jnp.zeros_like(x1), zp1)

    y_type = jax.ShapeDtypeStruct(list(ts.shape) + list(xz.shape), xz.dtype)
    yp_type = jax.ShapeDtypeStruct(list(ts.shape) + list(xzp.shape), xzp.dtype)

    xz, xzp = jax.pure_callback(
        callbacks.run_adjoint,
        (y_type, yp_type),
        (params, yfunc),
        ts[::-1],
        xz,
        xzp,
        vmap_method="sequential",
    )
    z = xz[:, x1.size :]

    return z[::-1]


@run_adjoint.defjvp
def run_adjoint_jvp(callbacks: SemiExplicitDAE, primals, tangents):
    yfunc, params, ts, x1, y1 = primals
    _, d_params, d_ts, _, d_z1 = tangents
    zp1 = callbacks.deriv_adj(params, ts[-1], x1, y1, yfunc)
    z1_a = jnp.zeros_like(y1)
    z1_y0 = d_z1
    z1_t0 = -zp1 * d_ts[-1]

    z1 = jnp.concatenate([y1, z1_a, z1_y0, z1_t0])
    xz = jnp.append(x1, z1)
    xzp = jnp.append(
        jnp.zeros_like(x1),
        callbacks.deriv_adj_ext((params, d_params, yfunc), ts[-1], x1, z1),
    )
    y_type = jax.ShapeDtypeStruct(list(ts.shape) + list(xz.shape), xz.dtype)
    yp_type = jax.ShapeDtypeStruct(list(ts.shape) + list(xzp.shape), xzp.dtype)

    xz, xzp = jax.pure_callback(
        callbacks.run_adjoint_extend,
        (y_type, yp_type),
        (params, d_params, yfunc),
        ts[::-1],
        xz,
        xzp,
        vmap_method="sequential",
    )
    x = xz[:, : x1.size]
    z = xz[:, x1.size :].reshape([ts.size, 4, y1.size])
    zp = xzp[:, x1.size :].reshape([ts.size, 4, y1.size])

    _, dz, _ = jax.vmap(
        _finalize_jvp, in_axes=(None, None, None, None, 0, 0, 0, 0, 0, None)
    )(
        callbacks.deriv_adj,
        callbacks.const_adj,
        params,
        d_params,
        ts,
        d_ts,
        x,
        z,
        zp,
        yfunc,
    )

    return z[::-1, 0, :], dz[::-1]


def daeint[Params, Var](
    params: Params,
    dae: SemiExplicitDAE,
    ts: Float[Array, " _"],
    xy0: Var,
    *,
    quad_order=4,
    options: dict = {},
    options_adj: dict = {},
):
    """
    Interface of SUNDIALS IDA solver for systems defined as

    Args:
    - options (dict): Additional options for the solver.
    """
    if quad_order not in [4]:
        raise NotImplementedError("quad_order must be 4.")

    x0, y0 = dae.partition(xy0)
    x, unravel_x = ravel_pytree(x0)
    y, unravel_y = ravel_pytree(y0)
    a, _ = ravel_pytree(params)

    x, y, yp = _daeint2(dae, a, ts, x, y)

    return Results(
        values=jax.vmap(lambda x, y: eqx.combine(unravel_x(x), unravel_y(y)))(x, y),
        derivatives=jax.vmap(lambda yp: unravel_y(yp))(yp),
    )
