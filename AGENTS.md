# IterFlow core repository

Follow the original HardFlow directory layout. Only Maze2D HardFlow and IterFlow
are in scope. Keep the IterFlow extension in hardflow/models_flow/iterflow and
the evaluation entry in run/eval.py. Do not add independent seed sweeps,
Projection, plotting or ablation experiment drivers unless requested.

Preserve Problem 6 / Algorithm 1: optimize predicted endpoint y, retain native
objective/constraints, constrain the affine actual successor F(y) to its history
ball, then set the successor to F(y*). No direct next-level ball on y or extra
post-solve projection.

Public defaults: configs/search.json, mixture search for attempts 1–150,
adaptive MH-style search from 151; outer objective C + 0.1E (contact weight zero; rollout statistics retained). Do not claim strict
stationary MCMC: archive changes and cached chain costs are inherited limitations.

Keep numerical admission distinct from execution safety. Goal arrival plus
avoidance of the two nominal added obstacles defines safe success. Never
silently rescore stored experiments. Keep weights, datasets and outputs ignored.
Run focused numerical tests and a small integration check after core changes.
