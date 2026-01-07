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

    try:
        xy, xyp = jax.pure_callback(
            _run_forward,
            (y_type, yp_type),
            params,
            ts,
            xy,
            xyp,
            vmap_method="sequential",
        )
    finally:
        residual._clear_cache()

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


def daeint[Params, Var](
    params: Params,
    deriv_fn: Callable[[Params, jax.Array, Var], Var],
    const_fn: Callable[[Params, jax.Array, Var], Any],
    ts: Float[Array, " _"],
    xy0: Var,
    *,
    quad_order=4,
    **options,
):
    """
    Interface of SUNDIALS IDA solver for systems defined as
    explicit ODE like y' = f(t, y),
    and semi-explicit DAE like y' = f(t, x, y), g(t, x, y) = 0.

    Parameters:
    - deriv_fn (Callable): A function that takes parameters, a coordinate `t`, and variables `x` and `y`.
      It returns the derivative `y'` of the differential variables `y`. The parameters and variables are pytrees.
      The return value `y'` is a pytree with the same structure as the input `x` and `y`,
      but with `None` in the positions corresponding to the algebraic variables `x`.

    - const_fn (Callable): A function that takes the same arguments as `deriv_fn` and returns the residuals
      of the constraints for the algebraic variables. The algebraic variables are computed such that the return value
      of `const_fn` becomes zero. If you want to solve an explicit ODE, you can pass a function that returns `None`.

    - options (dict): Additional options for the solver.
    """
    if quad_order not in [4]:
        raise NotImplementedError("quad_order must be 4.")

    yp0 = deriv_fn(params, ts[0], xy0)
    y0 = jax.tree.map(lambda u, v: u if v is not None else None, xy0, yp0)
    x0 = jax.tree.map(lambda u, v: u if v is None else None, xy0, yp0)
    try:
        utils.assert_trees_shape_equal(y0, yp0)
    except AssertionError as e:
        raise ValueError(
            "The shapes of initial conditions of differential variables and their derivative do not match. "
            "Check the initial conditions and deriv_fn()."
        ) from e
    x, unravel_x = ravel_pytree(x0)
    y, unravel_y = ravel_pytree(y0)
    a, unravel_a = ravel_pytree(params)

    def derivative(
        params_array: jax.Array, t: jax.Array, xarray: jax.Array, yarray: jax.Array
    ) -> jax.Array:
        params = unravel_a(params_array)
        x = unravel_x(xarray)
        y = unravel_y(yarray)
        xy = eqx.combine(x, y)
        yp = deriv_fn(params, t, xy)
        yparray, _ = ravel_pytree(yp)
        return yparray

    def constraint(
        params_array: jax.Array, t: jax.Array, xarray: jax.Array, yarray: jax.Array
    ) -> jax.Array:
        params = unravel_a(params_array)
        x = unravel_x(xarray)
        y = unravel_y(yarray)
        xy = eqx.combine(x, y)
        g = const_fn(params, t, xy)
        garray, _ = ravel_pytree(g)
        return garray

    def clear_cache():
        pass

    try:
        x, y, yp = _daeint(derivative, constraint, a, ts, x, y)
    except:
        clear_cache()
        raise

    return Results(
        values=jax.vmap(lambda x, y: eqx.combine(unravel_x(x), unravel_y(y)))(x, y),
        derivatives=jax.vmap(lambda yp: unravel_y(yp))(yp),
        clear_cache=clear_cache,
    )


class Results[U](NamedTuple):
    values: U
    derivatives: U
    clear_cache: Callable[[], None]
