import dataclasses
from typing import Callable, Any, NamedTuple, Protocol, runtime_checkable
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax.flatten_util import ravel_pytree
from sksundae._cy_ida import IDA as _IDA
import equinox as eqx
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from daex.utils import HermiteSpline
from daex import utils
from functools import partial


class Results[U](NamedTuple):
    values: U
    derivatives: U


class SemiExplicitDAE(eqx.Module):
    x_size: int
    partition: Callable
    derivative: Callable
    constraint: Callable
    deriv_fn: Callable
    const_fn: Callable
    resfn: Callable
    jacfn: Callable
    deriv_ext: Callable
    resfn_ext: Callable
    jacfn_ext: Callable
    deriv_adj: Callable
    const_adj: Callable
    resfn_adj: Callable
    jacfn_adj: Callable
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

    `deriv_fn` and `const_fn` should be wrapped with `jax.jit` and must not capture arrays
    as implicit inputs. Compiled code is cached by the identity of these functions,
    so pass the same jitted objects every time; wrapping them with `jax.jit` again
    on each call misses the cache.
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

    jacobian = jax.jit(jax.jacrev(residual, argnums=[2, 3], has_aux=False))

    def jacfn(t, y, yp, res, cj, JJ, userdata):
        (a,) = userdata
        t = jnp.asarray(t)
        y = jnp.asarray(y)
        yp = jnp.asarray(yp)
        dy, dyp = jacobian(a, t, y, yp)
        JJ[:, :] = np.asarray(dy + cj * dyp)

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

    jacobian_ext = jax.jit(jax.jacrev(residual_ext, argnums=[2, 3], has_aux=False))

    def jacfn_ext(t, y, yp, res, cj, JJ, userdata):
        a = jnp.asarray(userdata[0])
        da = jnp.asarray(userdata[1])
        t = jnp.asarray(t)
        y = jnp.asarray(y)
        yp = jnp.asarray(yp)
        dy, dyp = jacobian_ext((a, da), t, y, yp)
        JJ[:, :] = np.asarray(dy + cj * dyp)

    def da_fn(
        params: jax.Array,
        t: jax.Array,
        x: jax.Array,
        y: jax.Array,
        lam_g: jax.Array,
        lam_f: jax.Array,
    ) -> jax.Array:
        _, vjp_deriv = jax.vjp(deriv_fn, params, t, x, y)
        _, vjp_const = jax.vjp(const_fn, params, t, x, y)
        lam_f_dfda, _, _, _ = vjp_deriv(lam_f)
        lam_g_dgda, _, _, _ = vjp_const(lam_g)
        return lam_f_dfda - lam_g_dgda

    def deriv_adj(
        params: jax.Array,
        t: jax.Array,
        x: jax.Array,
        y: jax.Array,
        lam_g: jax.Array,
        lam_f: jax.Array,
    ):
        _, vjp_const = jax.vjp(const_fn, params, t, x, y)
        _, vjp_deriv = jax.vjp(deriv_fn, params, t, x, y)
        _, _, _, lam_f_dfdy = vjp_deriv(lam_f)
        _, _, _, lam_g_dgdy = vjp_const(lam_g)
        res = -lam_f_dfdy + lam_g_dgdy
        return res

    def const_adj(
        params: jax.Array,
        t: jax.Array,
        x: jax.Array,
        y: jax.Array,
        lam_g: jax.Array,
        lam_f: jax.Array,
    ):
        g, vjp_const = jax.vjp(const_fn, params, t, x, y)
        _, vjp_deriv = jax.vjp(deriv_fn, params, t, x, y)
        _, _, lam_f_dfdx, _ = vjp_deriv(lam_f)
        _, _, lam_g_dgdx, _ = vjp_const(lam_g)
        res = jnp.concatenate([g, lam_f_dfdx - lam_g_dgdx])
        return res

    @jax.jit
    def residual_adj(params_set, t, xy, xyp):
        # The adjoint DAE with lambda_f as the differential variable and
        # lambda_g and x as the algebraic variables:
        #   0 = g(t, x, y(t))
        #   0 = dfdx^T lambda_f - dgdx^T lambda_g
        #   0 = lambda_f' + dfdy^T lambda_f - dgdy^T lambda_g
        params, yfunc = params_set
        x = xy[:x_size]
        lam_g = xy[x_size : 2 * x_size]
        lam_f = xy[2 * x_size :]
        lam_fp = xyp[2 * x_size :]
        y = yfunc(t)
        deriv = deriv_adj(params, t, x, y, lam_g, lam_f)
        const = const_adj(params, t, x, y, lam_g, lam_f)
        res = jnp.concatenate([const, lam_fp - deriv])
        return res

    def resfn_adj(t, y, yp, res, userdata):
        t = jnp.asarray(t)
        y = jnp.asarray(y)
        yp = jnp.asarray(yp)
        res[:] = np.asarray(residual_adj(userdata, t, y, yp))

    jacobian_adj = jax.jit(jax.jacrev(residual_adj, argnums=[2, 3], has_aux=False))

    def jacfn_adj(t, y, yp, res, cj, JJ, userdata):
        t = jnp.asarray(t)
        y = jnp.asarray(y)
        yp = jnp.asarray(yp)
        dy, dyp = jacobian_adj(userdata, t, y, yp)
        JJ[:, :] = np.asarray(dy + cj * dyp)

    def clear_cache():
        residual._clear_cache()
        jacobian._clear_cache()
        residual_ext._clear_cache()
        jacobian_ext._clear_cache()
        residual_adj._clear_cache()
        jacobian_adj._clear_cache()

    return SemiExplicitDAE(
        x_size=x_size,
        partition=partition,
        derivative=derivative,
        constraint=constraint,
        deriv_fn=deriv_fn,
        const_fn=const_fn,
        resfn=resfn,
        jacfn=jacfn,
        deriv_ext=deriv_ext,
        resfn_ext=resfn_ext,
        jacfn_ext=jacfn_ext,
        deriv_adj=deriv_adj,
        const_adj=const_adj,
        resfn_adj=resfn_adj,
        jacfn_adj=jacfn_adj,
        da_fn=da_fn,
        _clear_cache=clear_cache,
    )


@dataclasses.dataclass(frozen=True)
class _UnravelEmpty:
    """Unravel function of a pytree without leaves, comparable by its structure."""

    treedef: Any

    def __call__(self, flat: jax.Array) -> Any:
        return jax.tree.unflatten(self.treedef, [])


def _ravel_pytree(pytree: Any) -> tuple[jax.Array, Callable[[jax.Array], Any]]:
    """
    `ravel_pytree` whose unravel function is equal for pytrees of the same structure.

    `ravel_pytree` returns an unravel function that compares by structure, except
    for a pytree without leaves, where it holds a fresh lambda on every call.
    """
    flat, unravel = ravel_pytree(pytree)
    leaves, treedef = jax.tree.flatten(pytree)
    if not leaves:
        return flat, _UnravelEmpty(treedef)
    return flat, unravel


@runtime_checkable
class _Model(Protocol):
    @property
    def x_size(self) -> int: ...
    def derivative(self, params: Any, t: Float[jax.Array, ""], xy: Any) -> Any: ...
    def constraint(self, params: Any, t: Float[jax.Array, ""], xy: Any) -> Any: ...
    def unravel_a(self, params_array: Float[jax.Array, " a_size"]) -> Any: ...
    def unravel_x(self, x_array: Float[jax.Array, " x_size"]) -> Any: ...
    def unravel_y(self, y_array: Float[jax.Array, " y_size"]) -> Any: ...


@dataclasses.dataclass(frozen=True)
class _ModelUSD:
    """
    User-defined functions and unravel functions of a DAE, passed to jitted
    functions as a static argument. Equal models share compiled code.
    """

    derivative: Callable
    constraint: Callable
    unravel_a: Callable
    unravel_x: Callable
    unravel_y: Callable
    x_size: int


class _Flattened(NamedTuple):
    model: _Model
    x: Float[Array, " x_size"]
    y: Float[Array, " y_size"]
    a: Float[Array, " a_size"]


def _make_model[Var](dae: SemiExplicitDAE, x0: Var, y0: Var, params: Any) -> _Flattened:
    x, unravel_x = _ravel_pytree(x0)
    y, unravel_y = _ravel_pytree(y0)
    a, unravel_a = _ravel_pytree(params)
    return _Flattened(
        model=_ModelUSD(
            derivative=dae.derivative,
            constraint=dae.constraint,
            unravel_a=unravel_a,
            unravel_x=unravel_x,
            unravel_y=unravel_y,
            x_size=len(x),
        ),
        x=x,
        y=y,
        a=a,
    )


def _deriv_fn(
    model: _Model,
    t: jax.Array,
    xarray: jax.Array,
    yarray: jax.Array,
    params_array: jax.Array,
) -> jax.Array:
    params = model.unravel_a(params_array)
    x = model.unravel_x(xarray)
    y = model.unravel_y(yarray)
    xy = eqx.combine(x, y)
    yp = model.derivative(params, t, xy)
    yparray, _ = ravel_pytree(yp)
    return yparray


def _const_fn(
    model: _Model,
    t: jax.Array,
    xarray: jax.Array,
    yarray: jax.Array,
    params_array: jax.Array,
) -> jax.Array:
    params = model.unravel_a(params_array)
    x = model.unravel_x(xarray)
    y = model.unravel_y(yarray)
    xy = eqx.combine(x, y)
    g = model.constraint(params, t, xy)
    garray, _ = ravel_pytree(g)
    return garray


@partial(jax.jit, static_argnums=0)
@jaxtyped(typechecker=typechecker)
def _res_fn(
    model: _Model,
    t: Float[jax.Array, ""],
    xy: Float[jax.Array, " var_size"],
    xyp: Float[jax.Array, " var_size"],
    a: Float[jax.Array, " a_size"],
) -> Float[jax.Array, " var_size"]:
    x, y = xy[: model.x_size], xy[model.x_size :]
    yp = xyp[model.x_size :]
    g = _const_fn(model, t, x, y, a)
    f = _deriv_fn(model, t, x, y, a)
    return jnp.concatenate([g, yp - f])


@partial(jax.jit, static_argnums=0)
@jaxtyped(typechecker=typechecker)
def _jac_fn(
    model: _Model,
    t: Float[jax.Array, ""],
    xy: Float[jax.Array, " var_size"],
    xyp: Float[jax.Array, " var_size"],
    cj: Float[jax.Array, ""],
    a: Float[jax.Array, " a_size"],
) -> Float[jax.Array, "var_size var_size"]:
    # d/dv res(xy + v, xyp + cj * v) = dres/dxy + cj * dres/dxyp, in one jacfwd.
    def shifted(v):
        return _res_fn(model, t, xy + v, xyp + cj * v, a)

    return jax.jacfwd(shifted)(jnp.zeros_like(xy))


class _AdjointState(NamedTuple):
    x: Float[Array, " x_size"]
    lam_g: Float[Array, " x_size"]
    lam_f: Float[Array, " y_size"]


class _AdjointParams(NamedTuple):
    params: Float[Array, " a_size"]
    yfunc: HermiteSpline


@dataclasses.dataclass(frozen=True)
class _AdjointModel:
    model: _Model
    unravel_x: Callable
    unravel_y: Callable
    unravel_a: Callable
    lam_g_size: int

    @property
    def x_size(self) -> int:
        return 2 * self.lam_g_size

    def derivative(
        self, params: _AdjointParams, t: Float[Array, ""], xy: _AdjointState
    ) -> _AdjointState:
        a, yfunc = params
        x, lam_g, lam_f = xy
        y = yfunc(t)

        _, vjp_deriv = jax.vjp(partial(_deriv_fn, self.model), t, x, y, a)
        _, vjp_const = jax.vjp(partial(_const_fn, self.model), t, x, y, a)
        _, _, lam_f_dfdy, _ = vjp_deriv(lam_f)
        _, _, lam_g_dgdy, _ = vjp_const(lam_g)
        return _AdjointState(
            x=None,
            lam_g=None,
            lam_f=-lam_f_dfdy + lam_g_dgdy,
        )

    def constraint(
        self, params: _AdjointParams, t: Float[Array, ""], xy: _AdjointState
    ):
        a, yfunc = params
        x, lam_g, lam_f = xy
        y = yfunc(t)

        _, vjp_deriv = jax.vjp(partial(_deriv_fn, self.model), t, x, y, a)
        g, vjp_const = jax.vjp(partial(_const_fn, self.model), t, x, y, a)
        _, lam_f_dfdx, _, _ = vjp_deriv(lam_f)
        _, lam_g_dgdx, _, _ = vjp_const(lam_g)
        return jnp.concatenate([g, lam_f_dfdx - lam_g_dgdx])


def _make_adjoint(
    model: _Model,
    x0: jax.Array,
    lam_g0: jax.Array,
    lam_f0: jax.Array,
    params: _AdjointParams,
) -> tuple[_AdjointModel, jax.Array, jax.Array, jax.Array]:
    x = _AdjointState(x=x0, lam_g=lam_g0, lam_f=None)
    y = _AdjointState(x=None, lam_g=None, lam_f=lam_f0)
    x, unravel_x = _ravel_pytree(x)
    y, unravel_y = _ravel_pytree(y)
    a, unravel_a = _ravel_pytree(params)
    return (
        _AdjointModel(
            model=model,
            unravel_x=unravel_x,
            unravel_y=unravel_y,
            unravel_a=unravel_a,
            lam_g_size=len(lam_g0),
        ),
        x,
        y,
        a,
    )


def _finalize_jvp(deriv_fn, const_fn, params, d_params, t, dt, x, y, yp, *args):
    y, z_a, z_y0, z_t0 = y
    yp, zp_a, zp_y0, zp_t0 = yp
    dfdt, dfdx, dfdy = jax.jacrev(deriv_fn, argnums=[1, 2, 3])(params, t, x, y, *args)
    dgda, dgdt, dgdx, dgdy = jax.jacrev(const_fn, argnums=[0, 1, 2, 3])(
        params, t, x, y, *args
    )
    lu_dgdx = jsp.linalg.lu_factor(dgdx)

    dy = z_a + z_y0 + z_t0 + yp * dt
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


@partial(jax.custom_vjp, nondiff_argnums=(0, 5, 6, 7))
def _daeint2(
    callbacks: tuple[SemiExplicitDAE, _Model],
    params: Float[Array, " a_size"],
    ts: Float[Array, " points"],
    x0: Float[Array, " x_size"],
    y0: Float[Array, " y_size"],
    quad_order: int,
    options: dict,
    options_sdj: dict,
) -> tuple[
    Float[Array, " x_size"],
    Float[Array, " y_size"],
    Float[Array, " y_size"],
]:
    """Perform DAE integration using IDA."""
    dae, _ = callbacks

    x, y, yp = run_forward(callbacks, params, ts, x0, y0, options)
    return x, y, yp


def _daeint_fwd2(
    callbacks: tuple[SemiExplicitDAE, _Model],
    params: Float[Array, " a_size"],
    ts: Float[Array, " points"],
    x0: Float[Array, " x_size"],
    y0: Float[Array, " y_size"],
    quad_order: int,
    options: dict,
    options_sdj: dict,
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
    dae, _ = callbacks
    n = (quad_order + 3) // 2
    ts, ws = utils.divide_intervals(ts[:-1], ts[1:], n=n)
    x, y, yp = run_forward(callbacks, params, ts, x0, y0, options)
    x1 = x[:: n - 1]
    y1 = y[:: n - 1]
    yp1 = yp[:: n - 1]
    return (x1, y1, yp1), (params, ts, ws, x, y, yp)


@partial(jax.custom_jvp, nondiff_argnums=(0, 5))
def run_forward(
    callbacks: tuple[SemiExplicitDAE, _Model], params, ts, x0, y0, options: dict
):
    _, model = callbacks
    yp0 = _deriv_fn(model, ts[0], x0, y0, params)
    xy = jnp.append(x0, y0)
    xyp = jnp.append(jnp.zeros_like(x0), yp0)
    y_type = jax.ShapeDtypeStruct(list(ts.shape) + list(xy.shape), xy.dtype)
    yp_type = jax.ShapeDtypeStruct(list(ts.shape) + list(xyp.shape), xyp.dtype)

    def _residual(t, xy, xyp, res, userdata):
        t = jnp.asarray(t)
        xy = jnp.asarray(xy)
        xyp = jnp.asarray(xyp)
        res[:] = _res_fn(model, t, xy, xyp, userdata[0])

    def _jacobian(t, xy, xyp, res, cj, JJ, userdata):
        t = jnp.asarray(t)
        xy = jnp.asarray(xy)
        xyp = jnp.asarray(xyp)
        cj = jnp.asarray(cj)
        JJ[:, :] = _jac_fn(model, t, xy, xyp, cj, userdata[0])

    def _call_ida(params: np.ndarray, ts: np.ndarray, y0: np.ndarray, yp0: np.ndarray):
        ida = _IDA(
            _residual,
            jacfn=_jacobian,
            userdata=(params,),
            algebraic_idx=np.arange(model.x_size),
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
        _call_ida,
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
def run_forward_jvp(
    callbacks: tuple[SemiExplicitDAE, _Model], options: dict, primals, tangents
):
    dae, _ = callbacks
    params, ts, x0, y0 = primals
    d_params, d_ts, _, d_y0 = tangents
    yp0 = dae.deriv_fn(params, ts[0], x0, y0)
    z_a = jnp.zeros_like(y0)
    z_y0 = d_y0
    z_t0 = -yp0 * d_ts[0]

    z0 = jnp.concatenate([y0, z_a, z_y0, z_t0])
    xy = jnp.append(x0, z0)
    xyp = jnp.append(
        jnp.zeros_like(x0), dae.deriv_ext((params, d_params), ts[0], x0, z0)
    )

    y_type = jax.ShapeDtypeStruct(list(ts.shape) + list(xy.shape), xy.dtype)
    yp_type = jax.ShapeDtypeStruct(list(ts.shape) + list(xyp.shape), xyp.dtype)

    def _call_ida(
        params: tuple[np.ndarray, np.ndarray],
        ts: np.ndarray,
        y0: np.ndarray,
        yp0: np.ndarray,
    ):
        ida = _IDA(
            dae.resfn_ext,
            jacfn=dae.jacfn_ext,
            userdata=params,
            algebraic_idx=np.arange(dae.x_size),
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
        _call_ida,
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
        dae.deriv_fn,
        dae.const_fn,
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
    callbacks: tuple[SemiExplicitDAE, _Model],
    quad_order: int,
    options: dict,
    options_adj: dict,
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
    dae, _ = callbacks
    params, ts, ws, x, y, yp = residuals
    wx, wy, wyp = cotangents
    n = (quad_order + 3) // 2
    points = wy.shape[0]
    tk = ts[:: n - 1]
    xk = x[:: n - 1]
    yk = y[:: n - 1]
    ypk = yp[:: n - 1]

    # Point multipliers mu_f, mu_g, mu_y at every ts[k]; mu_y at ts[0] is
    # replaced below by the multiplier of the initial condition.
    mu_f, mu_g, mu_y = jax.vmap(
        partial(_point_multipliers, callbacks), in_axes=(None, 0, 0, 0, 0, 0, 0)
    )(params, tk, xk, yk, wx, wy, wyp)

    def body(j, carry):
        k = points - 1 - j
        with jax.profiler.StepTraceAnnotation("daeint:backward_step", step_num=j):
            lam_next, dJda = carry
            # Jump condition: lambda_{f,k}(t_k) = mu_{y_k} + lambda_{f,k+1}(t_k)
            lam_f1 = mu_y[k] + lam_next
            start = (k - 1) * (n - 1)
            lam_f0, integral = _daeint_bwd_step2(
                callbacks,
                options_adj,
                params,
                jax.lax.dynamic_slice_in_dim(ts, start, n),
                ws[k - 1],
                jax.lax.dynamic_slice_in_dim(x, start, n),
                jax.lax.dynamic_slice_in_dim(y, start, n),
                jax.lax.dynamic_slice_in_dim(yp, start, n),
                lam_f1,
            )
            return lam_f0, dJda - integral

    # lambda_{f,K+1}(t_K) = 0
    lam_f_t0, dJda = jax.lax.fori_loop(
        0, points - 1, body, (jnp.zeros_like(wy[0]), jnp.zeros_like(params))
    )
    # mu_{y_0} = -lambda_{f,1}(t_0)
    mu_y = mu_y.at[0].set(-lam_f_t0)

    dJdy, dJdt, dJda_point = jax.vmap(
        partial(_point_vjp, callbacks), in_axes=(None, 0, 0, 0, 0, 0, 0, 0)
    )(params, tk, xk, yk, ypk, mu_f, mu_g, mu_y)
    dJda = dJda + jnp.sum(dJda_point, axis=0)

    # dJ/dy0 = w_{y,0} + mu_{y_0} - dfdy^T mu_{f_0} + dgdy^T mu_{g_0}
    dJdy0 = wy[0] + mu_y[0] + dJdy[0]

    return (dJda, dJdt, None, dJdy0)


_daeint2.defvjp(_daeint_fwd2, _daeint_bwd2)


def _point_multipliers(
    callbacks: tuple[SemiExplicitDAE, _Model],
    params: Float[Array, " a_size"],
    t: Float[Array, ""],
    x: Float[Array, " x_size"],
    y: Float[Array, " y_size"],
    wx: Float[Array, " x_size"],
    wy: Float[Array, " y_size"],
    wyp: Float[Array, " y_size"],
) -> tuple[Float[Array, " y_size"], Float[Array, " x_size"], Float[Array, " y_size"]]:
    """
    Multipliers of the point constraints at a time t_k with cotangents
    (wx, wy, wyp):
        mu_f = -wyp
        dgdx^T mu_g = -wx + dfdx^T mu_f
        mu_y = -wy + dfdy^T mu_f - dgdy^T mu_g
    """
    dae, _ = callbacks
    _, vjp_deriv = jax.vjp(dae.deriv_fn, params, t, x, y)
    _, vjp_const = jax.vjp(dae.const_fn, params, t, x, y)
    dgdx = jax.jacfwd(dae.const_fn, argnums=2)(params, t, x, y)

    mu_f = -wyp
    _, _, mu_f_dfdx, mu_f_dfdy = vjp_deriv(mu_f)
    mu_g = jnp.linalg.solve(dgdx.T, -wx + mu_f_dfdx)
    _, _, _, mu_g_dgdy = vjp_const(mu_g)
    mu_y = -wy + mu_f_dfdy - mu_g_dgdy
    return mu_f, mu_g, mu_y


def _point_vjp(
    callbacks: tuple[SemiExplicitDAE, _Model],
    params: Float[Array, " a_size"],
    t: Float[Array, ""],
    x: Float[Array, " x_size"],
    y: Float[Array, " y_size"],
    yp: Float[Array, " y_size"],
    mu_f: Float[Array, " y_size"],
    mu_g: Float[Array, " x_size"],
    mu_y: Float[Array, " y_size"],
) -> tuple[Float[Array, " y_size"], Float[Array, ""], Float[Array, " a_size"]]:
    """
    Point terms of the VJP at a time t_k:
        dJ/dt_k = -mu_y . yp - mu_f . dfdt + mu_g . dgdt
        -mu_f . dfda + mu_g . dgda  (the point term of dJ/da)
    """
    dae, _ = callbacks
    _, vjp_deriv = jax.vjp(dae.deriv_fn, params, t, x, y)
    _, vjp_const = jax.vjp(dae.const_fn, params, t, x, y)
    mu_f_dfda, mu_f_dfdt, _, mu_f_dfdy = vjp_deriv(mu_f)
    mu_g_dgda, mu_g_dgdt, _, mu_g_dgdy = vjp_const(mu_g)
    dJdy = -mu_f_dfdy + mu_g_dgdy
    dJdt = -jnp.dot(mu_y, yp) - mu_f_dfdt + mu_g_dgdt
    dJda = -mu_f_dfda + mu_g_dgda
    return dJdy, dJdt, dJda


def _daeint_bwd_step2(
    callbacks: tuple[SemiExplicitDAE, _Model],
    options: dict,
    params: Float[Array, " a_size"],
    ts: Float[Array, " quad_order"],
    ws: Float[Array, " quad_order"],
    x: Float[Array, "quad_order x_size"],
    y: Float[Array, "quad_order y_size"],
    yp: Float[Array, "quad_order y_size"],
    lam_f1: Float[Array, " y_size"],
) -> tuple[Float[Array, " y_size"], Float[Array, " a_size"]]:
    """
    Solve the adjoint DAE backward over one interval [ts[0], ts[-1]] from
    lambda_f(ts[-1]) = lam_f1, and return lambda_f(ts[0]) and the integral of
    lambda_f^T dfda - lambda_g^T dgda over the interval.
    """
    dae, model = callbacks
    yfunc = HermiteSpline(ts, y, yp)
    adj_params = _AdjointParams(params, yfunc)

    t1 = ts[-1]
    x1 = x[-1]
    y1 = y[-1]
    _, vjp_deriv = jax.vjp(partial(_deriv_fn, model), t1, x1, y1, params)
    dgdx = jax.jacfwd(_const_fn, argnums=2)(model, t1, x1, y1, params)
    lam_g1 = jnp.linalg.solve(dgdx.T, vjp_deriv(lam_f1)[1])

    adj_model, _x, _y, _a = _make_adjoint(model, x1, lam_g1, lam_f1, adj_params)
    _x, _y, _yp = run_forward((callbacks[0], adj_model), _a, ts[::-1], _x, _y, options)

    _, lam_g, _ = jax.vmap(adj_model.unravel_x)(_x[::-1])
    _, _, lam_f = jax.vmap(adj_model.unravel_y)(_y[::-1])

    with jax.profiler.TraceAnnotation("daeint:integrate_da"):
        integrand = jax.vmap(dae.da_fn, in_axes=(None, 0, 0, 0, 0, 0))(
            params, ts, x, y, lam_g, lam_f
        )
        integral = jnp.dot(ws, integrand)

    return lam_f[0], integral


def run_adjoint(
    callbacks: tuple[SemiExplicitDAE, _Model],
    yfunc,
    params,
    ts,
    x1,
    lam_f1,
    options: dict,
):
    dae, _ = callbacks
    t1 = ts[-1]
    y1 = yfunc(t1)
    _, vjp_deriv = jax.vjp(dae.deriv_fn, params, t1, x1, y1)
    dgdx = jax.jacfwd(dae.const_fn, argnums=2)(params, t1, x1, y1)
    lam_g1 = jnp.linalg.solve(dgdx.T, vjp_deriv(lam_f1)[2])
    lam_fp1 = dae.deriv_adj(params, t1, x1, y1, lam_g1, lam_f1)
    xz = jnp.concatenate([x1, lam_g1, lam_f1])
    xzp = jnp.concatenate([jnp.zeros_like(x1), jnp.zeros_like(lam_g1), lam_fp1])

    y_type = jax.ShapeDtypeStruct(list(ts.shape) + list(xz.shape), xz.dtype)
    yp_type = jax.ShapeDtypeStruct(list(ts.shape) + list(xzp.shape), xzp.dtype)

    def _call_ida(
        params: tuple[np.ndarray, HermiteSpline],
        ts: np.ndarray,
        y0: np.ndarray,
        yp0: np.ndarray,
    ):
        ida = _IDA(
            dae.resfn_adj,
            jacfn=dae.jacfn_adj,
            userdata=params,
            algebraic_idx=np.arange(2 * dae.x_size),
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

    xz, xzp = jax.pure_callback(
        _call_ida,
        (y_type, yp_type),
        (params, yfunc),
        ts[::-1],
        xz,
        xzp,
        vmap_method="sequential",
    )
    lam_g = xz[:, x1.size : 2 * x1.size]
    lam_f = xz[:, 2 * x1.size :]

    return lam_g[::-1], lam_f[::-1]


def daeint[Params, Var](
    params: Params,
    dae: SemiExplicitDAE,
    ts: Float[Array, " _"],
    xy0: Var,
    *,
    quad_order=5,
    options: dict = {},
    options_adj: dict = {},
) -> Results[Var]:
    """
    Interface of SUNDIALS IDA solver for systems defined as

    Args:
    - options (dict): Additional options for the solver.
    """
    if quad_order < 0:
        raise NotImplementedError("quad_order must be positive.")
    if quad_order % 2 == 0:
        raise NotImplementedError("quad_order must be odd.")

    x0, y0 = dae.partition(xy0)
    model, x, y, a = _make_model(dae, x0, y0, params)
    unravel_x = model.unravel_x
    unravel_y = model.unravel_y

    x, y, yp = _daeint2((dae, model), a, ts, x, y, quad_order, options, options_adj)

    with jax.profiler.TraceAnnotation("daeint:calc_dxdt"):

        def for_each(a, t, x, y, yp):
            dgdx = jax.jacfwd(dae.const_fn, argnums=2)(a, t, x, y)
            dxdt = -jnp.linalg.solve(
                dgdx,
                jax.jvp(
                    dae.const_fn,
                    (a, t, x, y),
                    (jnp.zeros_like(a), jnp.ones_like(t), jnp.zeros_like(x), yp),
                )[1],
            )
            return dxdt

        xp = jax.vmap(for_each, in_axes=(None, 0, 0, 0, 0))(a, ts, x, y, yp)

    return Results(
        values=jax.vmap(lambda x, y: eqx.combine(unravel_x(x), unravel_y(y)))(x, y),
        derivatives=jax.vmap(lambda xp, yp: eqx.combine(unravel_x(xp), unravel_y(yp)))(
            xp, yp
        ),
    )


class AdjointResult[Var](NamedTuple):
    """
    Multipliers of the Lagrangian L defined in the docstring of `adjoint`.

    - lambda_f: lambda_f(t), multiplier of y' = f on each interval.
    - lambda_g: lambda_g(t), multiplier of g = 0 on each interval.
    - mu_f: mu_{f_k}, multiplier of yp_k = f(t_k) at every t_k.
    - mu_g: mu_{g_k}, multiplier of g(t_k) = 0 at every t_k.
    - mu_y0: mu_{y_0}, multiplier of the initial condition y(t_0) = y0.
    """

    lambda_f: Var
    lambda_g: Var
    mu_f: Var
    mu_g: Var
    mu_y0: Var


def adjoint[Params, Var](
    params: Params,
    dae: SemiExplicitDAE,
    ts: Float[Array, " points"],
    solution: Results[Var],
    cotangent: Results[Var],
    *,
    options: dict = {},
) -> AdjointResult[Var]:
    """
    Compute the adjoint trajectory of a semi-explicit DAE.

    Given the equation system, a forward solution trajectory sampled at
    `ts`, and cotangents matching that trajectory (one entry per point,
    as in reverse-mode automatic differentiation), integrate the adjoint
    DAE backward.

    The Lagrangian

    Let t_k = ts[k] (k = 0..K, K = len(ts) - 1). The forward problem is
    y' = f(a, t, x, y), 0 = g(a, t, x, y) on [t_0, t_K] with y(t_0) = y0.
    The loss J sees the solution only through its values at the sample
    points, which are introduced as independent variables
    (x_k, y_k, yp_k) and tied to the trajectory by point constraints
    (y_0 is y0 itself). The cotangents are wx_k = dJ/dx_k,
    wy_k = dJ/dy_k, wyp_k = dJ/dyp_k.

        L(a, y0) = J(x_0..x_K, y_0..y_K, yp_0..yp_K)
            + sum_{k=0}^{K} mu_{y_k}^T (y_k - y(t_k))
            + sum_{k=0}^{K} mu_{f_k}^T (yp_k - f(a, t_k, x_k, y_k))
            + sum_{k=0}^{K} mu_{g_k}^T g(a, t_k, x_k, y_k)
            + sum_{k=0}^{K-1} int_{t_k}^{t_{k+1}}
                  [ lambda_f(t)^T (y'(t) - f(a, t, x(t), y(t)))
                    + lambda_g(t)^T g(a, t, x(t), y(t)) ] dt

    Making L stationary in every primal variable defines the multipliers:

        yp_k:  mu_{f_k} = -wyp_k
        x_k:   dgdx^T mu_{g_k} = -wx_k + dfdx^T mu_{f_k}
        y_k:   mu_{y_k} = -wy_k + dfdy^T mu_{f_k} - dgdy^T mu_{g_k}
               (Jacobians at (t_k, x_k, y_k))
        x(t), y(t) inside each interval -- the adjoint DAE:
               lambda_f' = -dfdy^T lambda_f + dgdy^T lambda_g
               0 = dfdx^T lambda_f - dgdx^T lambda_g
        y(t_k) (boundary terms of integrating lambda_f^T y' by parts):
               lambda_f(t_K+) = 0
               lambda_f(t_k-) = lambda_f(t_k+) + mu_{y_k}   (0 < k <= K)
               mu_{y_0} = -lambda_f(t_0+)

    With these multipliers the gradients of the loss are

        dJ/dy0 = wy_0 + mu_{y_0} - dfdy^T mu_{f_0} + dgdy^T mu_{g_0}
        dJ/da = sum_{k=0}^{K} (-dfda^T mu_{f_k} + dgda^T mu_{g_k})
            - sum_{k=0}^{K-1} int_{t_k}^{t_{k+1}}
                  (dfda^T lambda_f - dgda^T lambda_g) dt

    Returns:

    - `lambda_f`: lambda_f(t). Because of the jumps at the interior
      points of `ts`, it is stored per interval with shape
      `(len(ts) - 1, 2, ...)`: `lambda_f[k, 0]` is lambda_f(t_k+) and
      `lambda_f[k, 1]` is lambda_f(t_{k+1}-), the value after t_{k+1}'s
      own jump that seeds backward integration of the next, earlier
      interval.
    - `lambda_g`: lambda_g(t), with the same layout as `lambda_f`.
    - `mu_f`: mu_{f_k} at every point of `ts`, shape `(len(ts), ...)`.
    - `mu_g`: mu_{g_k} at every point of `ts`, shape `(len(ts), ...)`.
    - `mu_y0`: mu_{y_0}, the multiplier of the initial condition
      y(t_0) = y0.

    Args:
    - params (Params): Parameters, as passed to `daeint`.
    - dae (SemiExplicitDAE): The equation system, as returned by
      `def_semi_explicit_dae`.
    - ts (Array): Coordinates at which `solution`/`cotangent` are sampled.
    - solution (Results[Var]): The forward solution.
    - cotangent (Results[Var]): Cotangent matching `solution`.
    - options (dict): Additional options for the backward IDA solver,
      passed through to the adjoint DAE integration.
    """
    solution_values, solution_derivative = solution
    cotangent_values, cotangent_derivative = cotangent

    x0_sample, y0_sample = dae.partition(
        jax.tree.map(lambda leaf: leaf[0], solution_values)
    )
    model, _, _, a = _make_model(dae, x0_sample, y0_sample, params)
    unravel_x = model.unravel_x
    unravel_y = model.unravel_y

    def ravel_xy(xy: Var) -> tuple[jax.Array, jax.Array]:
        x0, y0 = dae.partition(xy)
        xarray, _ = ravel_pytree(x0)
        yarray, _ = ravel_pytree(y0)
        return xarray, yarray

    x, y = jax.vmap(ravel_xy)(solution_values)
    _, yp = jax.vmap(ravel_xy)(solution_derivative)
    wx, wy = jax.vmap(ravel_xy)(cotangent_values)
    _, wyp = jax.vmap(ravel_xy)(cotangent_derivative)

    points = ts.shape[0]
    ts_r = ts[::-1]
    x_r = x[::-1]
    y_r = y[::-1]
    yp_r = yp[::-1]

    # Point multipliers mu_f, mu_g, mu_y at every ts[k].
    mu_f, mu_g, mu_y = jax.vmap(
        partial(_point_multipliers, (dae, model)), in_axes=(None, 0, 0, 0, 0, 0, 0)
    )(a, ts, x, y, wx, wy, wyp)
    mu_y_r = mu_y[::-1]

    # lam_f_r[j] / lam_g_r[j] hold the values at both ends of the j-th
    # interval counted from the end, i.e. [ts[k], ts[k+1]] with k = points-2-j.
    lam_f_r = jnp.zeros((points - 1, 2, y.shape[1]), dtype=y.dtype)
    lam_g_r = jnp.zeros((points - 1, 2, x.shape[1]), dtype=x.dtype)

    def body(i, carry):
        lam_next, lam_g_r, lam_f_r = carry
        interval_ts = jnp.stack([ts_r[i], ts_r[i - 1]])
        interval_y = jnp.stack([y_r[i], y_r[i - 1]])
        interval_yp = jnp.stack([yp_r[i], yp_r[i - 1]])
        yfunc = HermiteSpline(interval_ts, interval_y, interval_yp)
        # Jump condition: lambda_f(ts[k]-) = mu_y at ts[k] + lambda_f(ts[k]+)
        lam_f1 = mu_y_r[i - 1] + lam_next
        lam_g, lam_f = run_adjoint(
            (dae, model), yfunc, a, interval_ts, x_r[i - 1], lam_f1, options
        )
        lam_g_r = lam_g_r.at[i - 1].set(lam_g)
        lam_f_r = lam_f_r.at[i - 1].set(lam_f)
        return lam_f[0], lam_g_r, lam_f_r

    # lambda_{f,K+1}(ts[-1]) = 0
    _, lam_g_r, lam_f_r = jax.lax.fori_loop(
        1, points, body, (jnp.zeros_like(wy[0]), lam_g_r, lam_f_r)
    )
    lam_g = lam_g_r[::-1]  # lam_g[k] = lambda_g at (ts[k]+, ts[k+1]-)
    lam_f = lam_f_r[::-1]  # lam_f[k] = lambda_f at (ts[k]+, ts[k+1]-)

    # mu_{y_0} = -lambda_{f,1}(ts[0])
    mu_y0 = -lam_f[0, 0]

    return AdjointResult(
        lambda_f=jax.vmap(jax.vmap(unravel_y))(lam_f),
        lambda_g=jax.vmap(jax.vmap(unravel_x))(lam_g),
        mu_f=jax.vmap(unravel_y)(mu_f),
        mu_g=jax.vmap(unravel_x)(mu_g),
        mu_y0=unravel_y(mu_y0),
    )
