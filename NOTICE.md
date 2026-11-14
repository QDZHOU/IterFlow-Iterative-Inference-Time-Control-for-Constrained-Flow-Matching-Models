# Attribution

HardFlow model, training, configuration, normalization, dynamics and surrogate
code are derived from the existing local HardFlow Maze2D checkout, associated
with **HardFlow: Hard-Constrained Sampling for Flow-Matching Models via Trajectory
Optimization** (Zeyang Li, Kaveh Alim, Navid Azizan; arXiv:2511.08425).

The IterFlow extension was extracted from the local `lmpc_hardflow` history-ball
implementation. Unused experimental drivers and alternate-method code have
been removed; the native HardFlow and history-ball numerical kernels remain.
The original repositories and their saved experiment results are unchanged.

The source HardFlow checkout had no top-level license. No new rights or license
are asserted for that upstream code. Native Maze2D retains its D4RL attribution
and Apache-2.0 license in `hardflow/envs/D4RL_LICENSE`.

Weights, datasets and local environments are not included. The local asset
links are ignored by Git.
