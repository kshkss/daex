# daex
Wrapper utilities for scikit-SUNDAE to be used on JAX ecosysytem

## Differentiation modes

`daeint(..., mode=...)` selects how first-order derivatives of the DAE
solution are computed:

- `mode="reverse"` (default): differentiate with `jax.grad`/`jax.vjp`.
  This matches the solver's original behavior.
- `mode="forward"`: differentiate with `jax.jvp`. `quad_order` and
  `options_adj` are ignored in this mode, since there is no backward
  (adjoint) pass.

Only first-order differentiation is guaranteed for either mode today.
Nesting differentiation transforms beyond first order (e.g.
`jax.grad(jax.grad(...))` or `jax.jvp(jax.jvp(...))`) is not supported and
will raise an error from JAX itself. Support for higher-order modes
(`reverse_forward`, `alternating`) is tracked separately.

# Tasks
## Tests
```
uv run pytest tests
```
