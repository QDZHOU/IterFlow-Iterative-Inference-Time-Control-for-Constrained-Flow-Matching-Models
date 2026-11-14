# Core validation — 2026-11-13

After restructuring to the original HardFlow layout and removing seed experiment,
Projection, plotting and ablation drivers:

- 34 focused tests passed.
- All relative imports resolve in `hardflow/models_flow/iterflow`.
- Original-flow and HardFlow baseline arrays match the pre-cleanup reference
  exactly: maximum absolute error is zero for initial noise, states, controls,
  natural references and physical final trajectory.
- Two real IterFlow attempts completed using the existing checkpoint and native
  MuJoCo environment. For this integration check only, the MCMC boundary was set
  to round 2 to exercise both phases. One candidate improved the incumbent;
  the second was rejected for higher cost. The normal default remains round 151.
- The inner surrogate, affine successor-ball constraint, hard constraints and
  outer cost are unchanged. The native policy no longer contains alternate
  Projection, Gradient Guidance or OC-Flow implementations.

Validation artifacts are stored outside the repository in the local temporary
directory. No historical experiment results or original source repositories were
modified. This is a functionality check, not a benchmark performance claim.
