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


class Results[U](NamedTuple):
    values: U
    derivatives: U
    clear_cache: Callable[[], None]


class IDA(eqx.Module):
    """
    Wrapper of SUNDIALS IDA solver.
    This solves semi-implicit DAE like y' = f(t, x, y), g(t, x, y) = 0.
    """

    deriv_fn: Callable[[Any, jax.Array, Any], Any]
    const_fn: Callable[[Any, jax.Array, Any], Any]
    options: dict

    def __init__(self, deriv_fn: Callable, const_fn: Callable, **options):
        self.deriv_fn = deriv_fn
        self.const_fn = const_fn
        self.options = options

    def solve[U](self, params: Any, ts: jax.Array, xy0: U) -> Results[U]:
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
        x_size = x.size
        y_size = y.size
        points = ts.shape[0]

        options["algebraic_idx"] = np.arange(x_size)

        def deriv_fn(
            params: Any, t: jax.Array, xarray: jax.Array, yarray: jax.Array
        ) -> jax.Array:
            x = unravel_x(xarray)
            y = unravel_y(yarray)
            xy = eqx.combine(x, y)
            yp = self.deriv_fn(params, t, xy)
            yparray, _ = ravel_pytree(yp)
            return yparray

        def const_fn(
            params: Any, t: jax.Array, xarray: jax.Array, yarray: jax.Array
        ) -> jax.Array:
            x = unravel_x(xarray)
            y = unravel_y(yarray)
            xy = eqx.combine(x, y)
            g = self.const_fn(params, t, xy)
            garray, _ = ravel_pytree(g)
            return garray

        def adjfn(
            params: Any,
            t: jax.Array,
            xarray: jax.Array,
            yarray: jax.Array,
            zarray: jax.Array,
        ):
            dfdx, dfdy = jax.jacfwd(deriv_fn, argnums=[2, 3])(params, t, xarray, yarray)
            dgdx, dgdy = jax.jacfwd(const_fn, argnums=[2, 3])(params, t, xarray, yarray)
            dfdz = dfdy - dfdx @ jnp.linalg.solve(dgdx, dgdy)
            dz = -dfdz.T @ zarray
            return dz

        @jax.jit
        def da_fn(
            params: Any,
            t: jax.Array,
            xarray: jax.Array,
            yarray: jax.Array,
            zarray: jax.Array,
        ) -> jax.Array:
            dfda, dfdx = jax.jacrev(deriv_fn, argnums=[0, 2])(params, t, xarray, yarray)
            dgda, dgdx = jax.jacrev(const_fn, argnums=[0, 2])(params, t, xarray, yarray)
            lu_dgdx = jsp.linalg.lu_factor(dgdx)
            dfda_x = jax.tree.map(
                lambda dgda0: dfdx @ jsp.linalg.lu_solve(lu_dgdx, dgda0), dgda
            )
            da = jax.tree.map(lambda a1, a2: (a1 - a2).T @ zarray, dfda, dfda_x)
            return da

        @jax.jit
        def resfn(
            params: Any,
            t: jax.Array,
            y: jax.Array,
            yp: jax.Array,
        ) -> tuple[jax.Array, None]:
            xarray = y[:x_size]
            yarray = y[x_size:]
            yp_est = yp[x_size:]
            yparray = deriv_fn(params, t, xarray, yarray)
            garray = const_fn(params, t, xarray, yarray)
            res = jnp.concatenate([yp_est - yparray, garray])
            return res, None

        @jax.jit
        def resfn_adj(
            params: Any,
            yfunc: Callable,
            t: jax.Array,
            y: jax.Array,
            yp: jax.Array,
        ) -> tuple[jax.Array, None]:
            xarray = y[:x_size]
            yarray = yfunc(t)
            zarray = y[x_size:]
            zp_est = yp[x_size:]
            garray = const_fn(params, t, xarray, yarray)
            zparray = adjfn(params, t, xarray, yarray, zarray)
            res = jnp.concatenate([garray, zp_est - zparray])
            return res, None

        def resfn_wrapper(t, y, yp, res, userdata):
            params = userdata
            t = jnp.asarray(t)
            y = jnp.asarray(y)
            yp = jnp.asarray(yp)
            res[:] = resfn(params, t, y, yp)[0]

        if "jacfn" not in self.options:

            def jacfn_wrapper(t, y, yp, res, cj, JJ, userdata):
                params = userdata
                t = jnp.asarray(t)
                y = jnp.asarray(y)
                yp = jnp.asarray(yp)
                (dy, dyp), _ = jax.jacfwd(resfn, argnums=[2, 3], has_aux=True)(
                    params, t, y, yp
                )
                JJ[:, :] = dy + cj * dyp

            self.options["jacfn"] = jacfn_wrapper

        def resfn_adj_wrapper(t, y, yp, res, userdata):
            params, yfunc = userdata
            t = jnp.asarray(t)
            y = jnp.asarray(y)
            yp = jnp.asarray(yp)
            res[:] = resfn_adj(params, yfunc, t, y, yp)[0]

        def jacfn_adj_wrapper(t, y, yp, res, cj, JJ, userdata):
            params, yfunc = userdata
            t = jnp.asarray(t)
            y = jnp.asarray(y)
            yp = jnp.asarray(yp)
            (dy, dyp), _ = jax.jacfwd(resfn_adj, argnums=[3, 4], has_aux=True)(
                params, yfunc, t, y, yp
            )
            JJ[:, :] = dy + cj * dyp

        options_ad = options.copy()
        options_ad["jacfn"] = jacfn_adj_wrapper
        options_ad["algebraic_idx"] = np.arange(x_size)

        def _run_forward(params: Any, ts: jax.Array, y0: jax.Array, yp0: jax.Array):
            ida = _IDA(resfn_wrapper, userdata=params, **options)
            results = ida.solve(ts, y0, yp0)
            if not results.success:
                raise RuntimeError(f"Adjoint IDA solver failed: {results.message}")
            if ts.shape[0] == 2:
                y = jnp.take(results.y, jnp.array([0, -1]), axis=0)
                yp = jnp.take(results.y, jnp.array([0, -1]), axis=0)
            else:
                y = results.y
                yp = results.yp
            return y, yp

        def _run_adjoint(
            params: Any, yfunc: Callable, ts: jax.Array, y0: jax.Array, yp0: jax.Array
        ):
            ida = _IDA(
                resfn_adj_wrapper,
                userdata=(params, yfunc),
                **options_ad,
            )
            results = ida.solve(ts, y0, yp0)
            if not results.success:
                raise RuntimeError(f"Adjoint IDA solver failed: {results.message}")
            return results.y

        @jax.custom_vjp
        def dae_solve(
            params: Any,
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

            xy = jnp.concatenate([x0, y0])
            xyp = jnp.concatenate([jnp.zeros_like(x0), yp0])
            y_type = jax.ShapeDtypeStruct(list(ts.shape) + list(xy.shape), xy.dtype)
            yp_type = jax.ShapeDtypeStruct(list(ts.shape) + list(xyp.shape), xyp.dtype)

            xy, xyp = jax.pure_callback(
                _run_forward, (y_type, yp_type), params, ts, xy, xyp
            )

            return xy[:, :x_size], xy[:, x_size:], xyp[:, x_size:]

        @jax.custom_vjp
        def dae_step(
            params: Any,
            t1: Float[Array, ""],
            t0: Float[Array, ""],
            x0: Float[Array, "x_size"],
            y0: Float[Array, "y_size"],
            yp0: Float[Array, "y_size"],
        ) -> tuple[
            Float[Array, "x_size"],
            Float[Array, "y_size"],
            Float[Array, "y_size"],
        ]:
            """Perform the first step of DAE integration using IDA solver."""

            # ida = _IDA(userdata=params, **args)
            ts = jnp.stack([t0, t1])
            xy = jnp.concatenate([x0, y0])
            xyp = jnp.concatenate([jnp.zeros_like(x0), yp0])
            y_type = jax.ShapeDtypeStruct([2] + list(xy.shape), xy.dtype)
            yp_type = jax.ShapeDtypeStruct([2] + list(xyp.shape), xyp.dtype)

            xy, xyp = jax.pure_callback(
                _run_forward, (y_type, yp_type), params, ts, xy, xyp
            )

            return xy[-1, :x_size], xy[-1, x_size:], xyp[-1, x_size:]

        def dae_solve_fwd(
            params: Any,
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
                Any,
                Float[Array, "3*points-2"],
                Float[Array, "3*points-2"],
                Float[Array, "3*points-2 x_size"],
                Float[Array, "3*points-2 y_size"],
                Float[Array, "3*points-2 y_size"],
            ],
        ]:
            ts, ws = utils.divide_intervals(ts[:-1], ts[1:], n=4)
            chex.assert_shape(ts, (3 * points - 2,))
            y = jnp.concatenate([x0, y0])
            yp = jnp.concatenate([jnp.zeros_like(x0), yp0])
            y_type = jax.ShapeDtypeStruct(list(ts.shape) + list(y.shape), y.dtype)
            yp_type = jax.ShapeDtypeStruct(list(ts.shape) + list(y.shape), yp.dtype)

            y, yp = jax.pure_callback(
                _run_forward, (y_type, yp_type), params, ts, y, yp
            )
            x1 = y[::3, :x_size]
            y1 = y[::3, x_size:]
            yp1 = yp[::3, x_size:]
            return (x1, y1, yp1), (
                params,
                ts,
                ws,
                y[:, :x_size],
                y[:, x_size:],
                yp[:, x_size:],
            )

        def dae_step_fwd(
            params: Any,
            t1: Float[Array, ""],
            t0: Float[Array, ""],
            x0: Float[Array, "x_size"],
            y0: Float[Array, "y_size"],
            yp0: Float[Array, "y_size"],
        ) -> tuple[
            tuple[
                Float[Array, "x_size"], Float[Array, "y_size"], Float[Array, "y_size"]
            ],
            tuple[
                Any,
                Float[Array, "4"],
                Float[Array, "4"],
                Float[Array, "4 x_size"],
                Float[Array, "4 y_size"],
                Float[Array, "4 y_size"],
            ],
        ]:
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
            ts = 0.5 * (t1 + t0) + 0.5 * (t1 - t0) * jnp.array(
                [-1.0, -0.447213595499957939282, 0.447213595499957939282, 1]
            )
            y = jnp.concatenate([x0, y0])
            yp = jnp.concatenate([jnp.zeros_like(x0), yp0])
            y_type = jax.ShapeDtypeStruct([4] + list(y.shape), y.dtype)
            yp_type = jax.ShapeDtypeStruct([4] + list(y.shape), yp.dtype)

            y, yp = jax.pure_callback(
                _run_forward, (y_type, yp_type), params, ts, y, yp
            )
            x1 = y[-1, :x_size]
            y1 = y[-1, x_size:]
            yp1 = yp[-1, x_size:]
            return (x1, y1, yp1), (
                params,
                ts,
                ws,
                y[:, :x_size],
                y[:, x_size:],
                yp[:, x_size:],
            )

        def dae_solve_bwd(
            residuals: tuple[
                Any,
                Float[Array, "3*points-2"],
                Float[Array, "3*points-2"],
                Float[Array, "3*points-2 x_size"],
                Float[Array, "3*points-2 y_size"],
                Float[Array, "3*points-2 y_size"],
            ],
            cotangents: tuple[
                Float[Array, "points x_size"],
                Float[Array, "points y_size"],
                Float[Array, "points y_size"],
            ],
        ) -> tuple[
            Any,  # parmas
            Float[Array, "points"],  # ts
            None,  # x0
            Float[Array, "y_size"],  # y0
            None,  # yp0
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

            dJdt = []
            dJdy0 = jnp.zeros_like(wy[0])
            dJda = jax.tree.map(lambda a: jnp.zeros_like(a), params)
            dJdt0_prev = 0.0
            n = 4 - 1
            for i in range(points - 1):
                dJda0, dJdt1, dJdt0, _, dJdy0, _ = dae_step_bwd(
                    residuals=(
                        params,
                        ts[i * n : i * n + 4][::-1],
                        ws[i * n : i * n + 4][::-1],
                        x[i * n : i * n + 4][::-1],
                        y[i * n : i * n + 4][::-1],
                        yp[i * n : i * n + 4][::-1],
                    ),
                    cotangents=(
                        wx[i],
                        wy[i] + dJdy0,
                        wyp[i],
                    ),
                )
                dJdt.append(dJdt1 + dJdt0_prev)
                dJdt0_prev = dJdt0
                dJda = jax.tree.map(lambda a1, a2: a1 + a2, dJda, dJda0)
            dJdt.append(dJdt0_prev)

            return (dJda, jnp.stack(dJdt)[::-1], None, dJdy0, None)

        def dae_step_bwd(
            residuals: tuple[
                Any,
                Float[Array, "4"],
                Float[Array, "4"],
                Float[Array, "4 x_size"],
                Float[Array, "4 y_size"],
                Float[Array, "4 y_size"],
            ],
            cotangents: tuple[
                Float[Array, "x_size"], Float[Array, "y_size"], Float[Array, "y_size"]
            ],
        ) -> tuple[
            Any,  # parmas
            Float[Array, ""],  # t1
            Float[Array, ""],  # t0
            None,  # x0
            Float[Array, "y_size"],  # y0
            None,  # yp0
        ]:
            params, ts, ws, x, y, yp = residuals
            wx, wy, wyp = cotangents
            t1 = ts[-1]
            x1 = x[-1]
            y1 = y[-1]
            yp1 = yp[-1]
            yfunc = HermiteSpline(ts, y, yp)

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
            dJda = jax.tree.map(
                lambda dfda0, dgda0: jnp.dot(wyp, dfda0)
                - jnp.dot(wxx, jsp.linalg.lu_solve(lu_dgdx, dgda0)),
                dfda,
                dgda,
            )

            z1 = (
                wy
                + jnp.dot(wyp, dfdy)
                - jnp.dot(wxx, jsp.linalg.lu_solve(lu_dgdx, dgdy))
            )
            dz1 = dfdy - dfdx @ jsp.linalg.lu_solve(lu_dgdx, dgdy)
            zp1 = -dz1.T @ z1
            z1 = jnp.concatenate([x1, z1])
            zp1 = jnp.concatenate([jnp.zeros_like(x1), zp1])

            z_type = jax.ShapeDtypeStruct([4] + list(z1.shape), z1.dtype)
            z = jax.pure_callback(
                _run_adjoint, z_type, params, yfunc, ts[::-1], z1, zp1
            )
            z = z[::-1, x_size:]
            # jax.debug.print("Adjoint: {z}", z=z)
            # jax.debug.print("Adjoint ts: {ts}", ts=ts)
            integral = jax.vmap(da_fn, in_axes=(None, 0, 0, 0, 0))(params, ts, x, y, z)
            dJda = jax.tree.map(lambda a1, a2: a1 + jnp.dot(ws, a2), dJda, integral)

            return (dJda, dJdt, -jnp.dot(z[0], yp[0]), None, z[0], None)

        dae_step.defvjp(dae_step_fwd, dae_step_bwd)
        dae_solve.defvjp(dae_solve_fwd, dae_solve_bwd)

        def clear_cache():
            resfn._clear_cache()
            resfn_adj._clear_cache()
            da_fn._clear_cache()

        try:
            x, y, yp = dae_solve(params, ts, x, y, yp)
        except:
            clear_cache()
            raise

        return Results(
            values=jax.vmap(lambda x, y: eqx.combine(unravel_x(x), unravel_y(y)))(x, y),
            derivatives=jax.vmap(lambda yp: unravel_y(yp))(yp),
            clear_cache=clear_cache,
        )
