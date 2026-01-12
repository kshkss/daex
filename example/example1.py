# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==6.0.0",
#     "chex==0.1.91",
#     "daex",
#     "jax==0.8.2",
#     "marimo>=0.19.2",
#     "numpy==2.4.0",
#     "polars[numpy,plot]==1.36.1",
# ]
#
# [tool.uv.sources]
# daex = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import numpy as np
    from daex.semi_explicit import daeint, def_semi_explicit_dae
    import equinox as eqx
    import polars as pl
    import altair as alt
    from typing import NamedTuple
    from jax.flatten_util import ravel_pytree
    return (
        NamedTuple,
        alt,
        daeint,
        def_semi_explicit_dae,
        eqx,
        jax,
        jnp,
        np,
        pl,
        ravel_pytree,
    )


@app.cell
def _(NamedTuple, daeint, def_semi_explicit_dae, eqx, jax, jnp):
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
        return stat.x ** 2 - jnp.sum(stat.y)

    def analytic(params: Params, ts: jax.Array, stat0: State) -> State:
        def result(params: Params, t: jax.Array, t0: jax.Array, stat0: State):
            x = 0.25 * params.a * (t**2 - t0 ** 2) + jnp.sqrt(stat0.y)
            return State(x=x, y=x**2)

        u = jax.vmap(result, in_axes=(None, 0, None, None))(params, ts, ts[0], stat0)
        v = jax.vmap(jax.jacfwd(result, argnums=1), in_axes=(None, 0, None, None))(params, ts, ts[0], stat0)

        return u, eqx.tree_at(lambda v: v.x, v, replace=None)

    params = Params(a=jnp.array(1.0))
    y0 = State(y=jnp.array(2.0), x=jnp.sqrt(2.0))
    dae = def_semi_explicit_dae(derivative, constraint, params, jnp.array(0.0), y0)

    u, v = daeint(params, dae, jnp.linspace(0, 1, 11), y0)
    return analytic, dae, params, u, v, y0


@app.cell
def _(analytic, jnp, params, y0):
    analytic(params, jnp.linspace(0, 1, 11), y0)
    return


@app.cell
def _(u):
    u
    return


@app.cell
def _(v):
    v
    return


@app.cell
def _(analytic, jax, jnp, params, u, y0):
    _u, _v = analytic(params, jnp.linspace(0, 1, 11), y0)
    jax.tree.map(lambda a, b: jnp.allclose(a, b, 1e-4, 1e-4), u, _u)
    return


@app.cell
def _(analytic, jax, jnp, params, v, y0):
    _u, _v = analytic(params, jnp.linspace(0, 1, 11), y0)
    jax.tree.map(lambda a, b: jnp.allclose(a, b, 1e-4, 1e-4), v, _v)
    return


@app.cell
def _(analytic, jnp, np, params, pl, u, y0):
    _acc, _ = analytic(params, jnp.linspace(0, 1, 11), y0)

    df = pl.DataFrame(dict(
        t=np.linspace(0, 1, 11),
        x=np.array(u.x),
        y=np.array(u.y),
    )).with_columns(
        x_acc=np.array(_acc.x),
        y_acc=np.array(_acc.y),
    )
    df
    return (df,)


@app.cell
def _(alt, df):
    # replace _df with your data source
    _chart = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x=alt.X(field='t', type='quantitative'),
            y=alt.Y(field='y', type='quantitative', aggregate='mean'),
            tooltip=[
                alt.Tooltip(field='t', format=',.2f'),
                alt.Tooltip(field='y', aggregate='mean', format=',.2f')
            ]
        )
        .properties(
            height=290,
            width='container',
        )
    )
    _chart2 = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(field='t', type='quantitative'),
            y=alt.Y(field='y_acc', type='quantitative', aggregate='mean'),
            tooltip=[
                alt.Tooltip(field='t', format=',.2f'),
                alt.Tooltip(field='y', aggregate='mean', format=',.2f')
            ]
        )
        .properties(
            height=290,
            width='container',
        )
    )
    _chart + _chart2
    return


@app.cell
def _(alt, df):
    # replace _df with your data source
    _chart = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x=alt.X(field='t', type='quantitative'),
            y=alt.Y(field='x', type='quantitative', aggregate='mean'),
            tooltip=[
                alt.Tooltip(field='t', format=',.2f'),
                alt.Tooltip(field='y', aggregate='mean', format=',.2f')
            ]
        )
        .properties(
            height=290,
            width='container',
        )
    )
    _chart2 = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(field='t', type='quantitative'),
            y=alt.Y(field='x_acc', type='quantitative', aggregate='mean'),
            tooltip=[
                alt.Tooltip(field='t', format=',.2f'),
                alt.Tooltip(field='y', aggregate='mean', format=',.2f')
            ]
        )
        .properties(
            height=290,
            width='container',
        )
    )
    _chart + _chart2
    return


@app.cell
def _(dae, daeint, jax, jnp, params, y0):
    def loss1(params, t, y0):
        u, v = daeint(params, dae, t, y0)
        return u.y[-1]

    jax.grad(loss1, argnums=[0, 1, 2])(params, jnp.linspace(-0.5, 1, 11), y0)
    return (loss1,)


@app.cell
def _(analytic, jax, jnp, params, y0):
    def loss1_acc(params, t, y0):
        u, v = analytic(params, t, y0)
        return u.y[-1]

    jax.grad(loss1_acc, argnums=[0, 1, 2])(params, jnp.linspace(-0.5, 1, 11), y0)
    return (loss1_acc,)


@app.cell
def _(jax, jnp, loss1, loss1_acc, params, y0):
    _u = jax.grad(loss1_acc, argnums=[0, 1, 2])(params, jnp.linspace(-0.5, 1, 11), y0)
    _v = jax.grad(loss1, argnums=[0, 1, 2])(params, jnp.linspace(-0.5, 1, 11), y0)
    jax.tree.map(lambda a, b: jnp.allclose(a, b, 1e-4, 1e-4), _u, _v)
    return


@app.cell
def _(dae, daeint, jax, jnp, params, y0):
    def loss2(params, t, y0):
        u, v = daeint(params, dae, t, y0)
        return u.x[-1]

    jax.grad(loss2, argnums=[0, 1, 2])(params, jnp.linspace(-0.5, 1, 11), y0)
    return (loss2,)


@app.cell
def _(analytic, jax, jnp, params, y0):
    def loss2_acc(params, t, y0):
        u, v = analytic(params, t, y0)
        return u.x[-1]

    jax.grad(loss2_acc, argnums=[0, 1, 2])(params, jnp.linspace(-0.5, 1, 11), y0)
    return (loss2_acc,)


@app.cell
def _(jax, jnp, loss2, loss2_acc, params, y0):
    _u = jax.grad(loss2_acc, argnums=[0, 1, 2])(params, jnp.linspace(-0.5, 1, 11), y0)
    _v = jax.grad(loss2, argnums=[0, 1, 2])(params, jnp.linspace(-0.5, 1, 11), y0)
    jax.tree.map(lambda a, b: jnp.allclose(a, b, 1e-4, 1e-4), _u, _v)
    return


@app.cell
def _(dae, daeint, jax, jnp, params, y0):
    def loss3(params, t, y0):
        u, v = daeint(params, dae, t, y0)
        return v.y[-1]

    jax.grad(loss3, argnums=[0, 1, 2])(params, jnp.linspace(-0.5, 1, 11), y0)
    return (loss3,)


@app.cell
def _(analytic, jax, jnp, params, y0):
    def loss3_acc(params, t, y0):
        u, v = analytic(params, t, y0)
        return v.y[-1]

    jax.grad(loss3_acc, argnums=[0, 1, 2])(params, jnp.linspace(-0.5, 1, 11), y0)
    return (loss3_acc,)


@app.cell
def _(jax, jnp, loss3, loss3_acc, params, y0):
    _u = jax.grad(loss3_acc, argnums=[0, 1, 2])(params, jnp.linspace(-0.5, 1, 11), y0)
    _v = jax.grad(loss3, argnums=[0, 1, 2])(params, jnp.linspace(-0.5, 1, 11), y0)
    jax.tree.map(lambda a, b: jnp.allclose(a, b, 1e-4, 1e-4), _u, _v)
    return


@app.cell
def _(jax, np):
    def hvp(f, primals, tangents):
      return jax.jvp(jax.grad(f, argnums=np.arange(len(primals))), primals, tangents)[1]
    return (hvp,)


@app.cell
def _(hvp, jnp, loss1, params, ravel_pytree, y0):
    _primals = (params, jnp.linspace(-0.5, 1, 11), y0)
    _x, _unravel = ravel_pytree(_primals)
    _tangents = _unravel(jnp.zeros_like(_x).at[0].set(1.0))

    hvp(loss1, _primals, _tangents)
    return


@app.cell
def _(hvp, jnp, loss1_acc, params, ravel_pytree, y0):
    _primals = (params, jnp.linspace(-0.5, 1, 11), y0)
    _x, _unravel = ravel_pytree(_primals)
    _tangents = _unravel(jnp.zeros_like(_x).at[0].set(1.0))

    hvp(loss1_acc, _primals, _tangents)
    return


@app.cell
def _(hvp, jax, jnp, loss1, loss1_acc, params, ravel_pytree, y0):
    _primals = (params, jnp.linspace(-0.5, 1, 11), y0)
    _x, _unravel = ravel_pytree(_primals)
    _tangents = _unravel(jnp.zeros_like(_x).at[0].set(1.0))

    jax.tree.map(lambda a, b: jnp.allclose(a, b, 1e-4, 1e-4), hvp(loss1, _primals, _tangents), hvp(loss1_acc, _primals, _tangents))
    return


@app.cell
def _(hvp, jnp, loss1_acc, params, ravel_pytree, y0):
    _primals = (params, jnp.linspace(-0.5, 1, 11), y0)
    _x, _unravel = ravel_pytree(_primals)
    _tangents = _unravel(jnp.zeros_like(_x).at[1].set(1.0))

    hvp(loss1_acc, _primals, _tangents)
    return


@app.cell
def _(hvp, jnp, loss1, params, ravel_pytree, y0):
    _primals = (params, jnp.linspace(-0.5, 1, 11), y0)
    _x, _unravel = ravel_pytree(_primals)
    _tangents = _unravel(jnp.zeros_like(_x).at[1].set(1.0))

    hvp(loss1, _primals, _tangents)
    return


@app.cell
def _(hvp, jax, jnp, loss1, loss1_acc, params, ravel_pytree, y0):
    _primals = (params, jnp.linspace(-0.5, 1, 11), y0)
    _x, _unravel = ravel_pytree(_primals)
    _tangents = _unravel(jnp.zeros_like(_x).at[1].set(1.0))

    jax.tree.map(lambda a, b: jnp.allclose(a, b, 1e-4, 1e-4), hvp(loss1, _primals, _tangents), hvp(loss1_acc, _primals, _tangents))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
