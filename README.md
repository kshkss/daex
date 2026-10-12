# daex
Wrapper utilities for scikit-SUNDAE to be used on JAX ecosysytem

## Differentiation modes
`daeint` takes a `mode` argument that selects which differentiation rule
(forward = JVP, reverse = VJP) is used at each order of differentiation.

```python
u, du = daeint(params, dae, ts, xy0, mode="reverse_forward")
```

| `mode` | 1st order | 2nd order | 3rd order | higher orders |
|---|---|---|---|---|
| `"forward"` | forward | forward | forward | forward |
| `"reverse"` (default) | reverse | reverse | reverse | reverse |
| `"reverse_forward"` | reverse | forward | forward | forward |
| `"alternating"` | reverse | forward | reverse | alternately forward and reverse |

Apply JAX transformations in the order that the mode supports, innermost first.
For example, with `mode="reverse_forward"`, a Hessian-vector product is
`jax.jvp(jax.grad(loss), ...)`. Other orders may be refused by JAX as
unsupported. Any other value of `mode` raises `ValueError`.

The algebraic variables in `xy0` must satisfy the constraint at `ts[0]`.
Their tangents are ignored, since they are determined by the differential
variables, `ts[0]` and `params`.

# Tasks
## Tests
```
uv run pytest tests
```
