# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==6.0.0",
#     "chex==0.1.91",
#     "daex",
#     "jax==0.8.2",
#     "numpy==2.4.0",
#     "polars[numpy,plot]==1.36.1",
# ]
#
# [tool.uv.sources]
# daex = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import numpy as np
    from daex import IDA
    import equinox as eqx
    import polars as pl
    import altair as alt
    from typing import NamedTuple
    return IDA, NamedTuple, alt, jax, jnp, np, pl


@app.cell
def _(IDA, NamedTuple, jax, jnp):
    class State(NamedTuple):
        x: jax.Array
        y: jax.Array

    class Params(NamedTuple):
        a: jax.Array

    def derivative(params: Params, t: jax.Array, stat: State) -> State:
        return State(
            x=None,
            y=stat.y * params.a,
        )

    def constraint(params: Params, t: jax.Array, stat: State) -> jax.Array:
        return stat.x ** 2 - stat.y

    solver = IDA(deriv_fn=derivative, const_fn=constraint)
    params = Params(a=jnp.array(1.0))
    u, v = solver.solve(params, jnp.linspace(0, 1, 11), State(y=jnp.array(1.0), x=jnp.array(1.0)))
    return State, derivative, params, solver, u, v


@app.cell
def _(u):
    u
    return


@app.cell
def _(v):
    v
    return


@app.cell
def _(State, jnp):
    _x = State(y=jnp.ones(1), x=jnp.ones(1))
    _x
    return


@app.cell
def _(State, derivative, jnp, params):
    _x = State(y=jnp.array(1.0), x=jnp.array(1.0))
    derivative(params, jnp.array(1.0), _x)
    return


@app.cell
def _(np, params, pl, u):
    df = pl.DataFrame(dict(
        t=np.linspace(0, 1, 11),
        x=np.array(u.x),
        y=np.array(u.y),
    )).with_columns(
        y_acc=np.exp(pl.col.t*params.a),
    ).with_columns(
        x_acc=np.sqrt(pl.col.y_acc),
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
def _(State, jax, jnp, params, solver):
    def loss(params, t, y0):
        u, v = solver.solve(params, t, State(y=y0, x=jnp.sqrt(y0)))
        return u.y[-1][0]

    jax.value_and_grad(loss, argnums=[0, 1, 2])(params, jnp.linspace(0, 1, 11), jnp.array(1.0))
    return


@app.cell
def _(State, jax, jnp, params, solver):
    def loss2(params, t, y0):
        u, v = solver.solve(params, t, State(y=y0, x=jnp.sqrt(y0)))
        return u.x[-1]

    jax.value_and_grad(loss2, argnums=[0, 1, 2])(params, jnp.linspace(0, 1, 11), jnp.array(1.0))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
