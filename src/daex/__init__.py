from typing import Callable, Any, NamedTuple
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax.flatten_util import ravel_pytree
from sksundae._cy_ida import IDA as _IDA
import equinox as eqx
from jaxtyping import Array, Float
import chex
from daex.utils import HermiteSpline
from daex import utils


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


class Results[U](NamedTuple):
    values: U
    derivatives: U
    clear_cache: Callable[[], None]


class SemiExplicitDAE[Params, Var](eqx.Module):
    """
    INterface of SUNDIALS IDA solver for systems defined as
    explicit ODE like y' = f(t, y),
    and semi-explicit DAE like y' = f(t, x, y), g(t, x, y) = 0.
    """

    deriv_fn: Callable[[Params, jax.Array, Var], Var]
    const_fn: Callable[[Params, jax.Array, Var], Any]
    quad_order: int
    options: dict

    def __init__(
        self, deriv_fn: Callable, const_fn: Callable, *, quad_order=4, **options
    ):
        """
        Initialize the SemiExplicitDAE class.

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
        self.deriv_fn = deriv_fn
        self.const_fn = const_fn
        self.options = options

        if quad_order not in [4]:
            raise NotImplementedError("quad_order must be 4.")
        self.quad_order = quad_order

    def solve(self, params: Params, ts: jax.Array, xy0: Var) -> Results[Var]:
        options = self.options.copy()
        yp0 = self.deriv_fn(params, ts[0], xy0)
        x0 = jax.tree.map(lambda xy, yp: xy if yp is None else None, xy0, yp0)
        y0 = jax.tree.map(lambda xy, yp: xy if yp is not None else None, xy0, yp0)
        try:
            utils.assert_trees_shape_equal(y0, yp0)
        except AssertionError as e:
            raise ValueError(
                "The shapes of initial conditions of differential variables and their derivative do not match. "
                "Check the initial conditions and deriv_fn()."
            ) from e
        x, unravel_x = ravel_pytree(x0)
        y, unravel_y = ravel_pytree(y0)
        yp, _ = ravel_pytree(yp0)
        a, unravel_a = ravel_pytree(params)
        x_size = x.size
        y_size = y.size
        a_size = a.size
        points = ts.shape[0]
        n_intervals = points - 1
        quad_order = self.quad_order
        n_split = quad_order - 1
        interpolated = n_split * n_intervals + 1

        def deriv_fn(
            params_array: jax.Array, t: jax.Array, xarray: jax.Array, yarray: jax.Array
        ) -> jax.Array:
            params = unravel_a(params_array)
            x = unravel_x(xarray)
            y = unravel_y(yarray)
            xy = eqx.combine(x, y)
            yp = self.deriv_fn(params, t, xy)
            yparray, _ = ravel_pytree(yp)
            return yparray

        def const_fn(
            params_array: jax.Array, t: jax.Array, xarray: jax.Array, yarray: jax.Array
        ) -> jax.Array:
            params = unravel_a(params_array)
            x = unravel_x(xarray)
            y = unravel_y(yarray)
            xy = eqx.combine(x, y)
            g = self.const_fn(params, t, xy)
            garray, _ = ravel_pytree(g)
            return garray

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

        @jax.jit
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

        @jax.custom_vjp
        def dae_solve(
            params: Float[Array, "a_size"],
            ts: Float[Array, "points"],
            x0: Float[Array, "x_size"],
            y0: Float[Array, "y_size"],
            yp0: Float[Array, "y_size"],
        ) -> tuple[
            Float[Array, "x_size"],
            Float[Array, "y_size"],
            Float[Array, "y_size"],
        ]:
            """Perform the first step of DAE integration using IDA solver."""

            return _call_ida(deriv_fn, const_fn, params, ts, x0, y0)

        def dae_solve_fwd(
            params: Float[Array, "a_size"],
            ts: Float[Array, "points"],
            x0: Float[Array, "x_size"],
            y0: Float[Array, "y_size"],
            yp0: Float[Array, "y_size"],
        ) -> tuple[
            tuple[
                Float[Array, "points x_size"],
                Float[Array, "points y_size"],
                Float[Array, "points y_size"],
            ],
            tuple[
                Float[Array, "a_size"],
                Float[Array, "interpolated"],
                Float[Array, "n_intervals quad_order"],
                Float[Array, "interpolated x_size"],
                Float[Array, "interpolated y_size"],
                Float[Array, "interpolated y_size"],
            ],
        ]:
            ts, ws = utils.divide_intervals(ts[:-1], ts[1:], n=4)
            chex.assert_shape(ts, (interpolated,))
            chex.assert_shape(ws, (n_intervals, quad_order))

            x, y, yp = _call_ida(deriv_fn, const_fn, params, ts, x0, y0)
            x1 = x[::3]
            y1 = y[::3]
            yp1 = yp[::3]
            return (x1, y1, yp1), (params, ts, ws, x, y, yp)

        def dae_solve_bwd(
            residuals: tuple[
                Float[Array, "a_size"],
                Float[Array, "interpolated"],
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
            Float[Array, "a_size"],  # parmas
            Float[Array, "points"],  # ts
            None,  # x0
            Float[Array, "y_size"],  # y0
            None,  # yp0
        ]:
            params, ts, ws, x, y, yp = residuals
            wx, wy, wyp = cotangents
            chex.assert_shape(ts, (interpolated,))
            chex.assert_shape(ws, (n_intervals, quad_order))
            chex.assert_shape(x, (interpolated, x_size))
            chex.assert_shape(y, (interpolated, y_size))
            chex.assert_shape(yp, (interpolated, y_size))
            chex.assert_shape(wx, (points, x_size))
            chex.assert_shape(wy, (points, y_size))
            chex.assert_shape(wyp, (points, y_size))
            ts = ts[::-1]
            ws = ws[::-1]
            x = x[::-1]
            y = y[::-1]
            yp = yp[::-1]
            wx = wx[::-1]
            wy = wy[::-1]
            wyp = wyp[::-1]

            def body(i, carry):
                dJda, dJdt0_prev, dJdt, dJdy0 = carry
                dJda0, dJdt1, dJdt0, _, dJdy0, _ = dae_step_bwd(
                    residuals=(
                        params,
                        jax.lax.dynamic_slice_in_dim(
                            ts, i * (quad_order - 1), quad_order
                        )[::-1],
                        ws[i],
                        jax.lax.dynamic_slice_in_dim(
                            x, i * (quad_order - 1), quad_order
                        )[::-1],
                        jax.lax.dynamic_slice_in_dim(
                            y, i * (quad_order - 1), quad_order
                        )[::-1],
                        jax.lax.dynamic_slice_in_dim(
                            yp, i * (quad_order - 1), quad_order
                        )[::-1],
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

            return (dJda, dJdt[::-1], None, dJdy0, None)

        def dae_step_bwd(
            residuals: tuple[
                Float[Array, "a_size"],
                Float[Array, "quad_order"],
                Float[Array, "quad_order"],
                Float[Array, "quad_order x_size"],
                Float[Array, "quad_order y_size"],
                Float[Array, "quad_order y_size"],
            ],
            cotangents: tuple[
                Float[Array, "x_size"], Float[Array, "y_size"], Float[Array, "y_size"]
            ],
        ) -> tuple[
            Float[Array, "a_size"],  # parmas
            Float[Array, ""],  # t1
            Float[Array, ""],  # t0
            None,  # x0
            Float[Array, "y_size"],  # y0
            None,  # yp0
        ]:
            params, ts, ws, x, y, yp = residuals
            wx, wy, wyp = cotangents
            chex.assert_shape(ts, (quad_order,))
            chex.assert_shape(ws, (quad_order,))
            chex.assert_shape(x, (quad_order, x_size))
            chex.assert_shape(y, (quad_order, y_size))
            chex.assert_shape(yp, (quad_order, y_size))
            chex.assert_shape(wx, (x_size,))
            chex.assert_shape(wy, (y_size,))
            chex.assert_shape(wyp, (y_size,))
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
            z1 = (
                wy
                + jnp.dot(wyp, dfdy)
                - jnp.dot(wxx, jsp.linalg.lu_solve(lu_dgdx, dgdy))
            )

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

            return (dJda, dJdt, -jnp.dot(z[0], yp[0]), None, z[0], None)

        dae_solve.defvjp(dae_solve_fwd, dae_solve_bwd)

        def clear_cache():
            da_fn._clear_cache()

        try:
            x, y, yp = dae_solve(a, ts, x, y, yp)
        except:
            clear_cache()
            raise

        return Results(
            values=jax.vmap(lambda x, y: eqx.combine(unravel_x(x), unravel_y(y)))(x, y),
            derivatives=jax.vmap(lambda yp: unravel_y(yp))(yp),
            clear_cache=clear_cache,
        )
