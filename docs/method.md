# IterFlow for Maze2D

Let `x_i` be the free coordinates of the full normalized planning tensor at
noise level `t_i`; fixed endpoint states are reinserted exactly. With model
velocity `v`, the native natural successor and predicted endpoint are

\[
 r_i=x_i+\Delta t_i v(x_i,t_i),\qquad
 \hat y_i=r_i+(1-t_{i+1})v(r_i,t_{i+1}).
\]

HardFlow's Algorithm 1 affine map is

\[
 F_{t_{i+1}}(y)=r_i+t_{i+1}(y-\hat y_i).
\]

The optimization variable remains the predicted endpoint `y`. On controlled
tail steps, IterFlow solves the existing Problem 6 surrogate with one additional
history-ball constraint:

\[
\begin{aligned}
\min_y\quad &C(y)+\frac{1}{2}t_{i+1}^2\|y-\hat y_i\|_2^2\\
\text{s.t.}\quad &g_{\mathrm{HF}}(y)\leq0,\quad h_{\mathrm{HF}}(y)=0,\\
&\|P F_{t_{i+1}}(y)-c_{i+1}\|_2\leq\rho_{i+1}+\epsilon_{i+1}.
\end{aligned}
\]

`P` selects the free dimensions used by the archive. The implementation already
works in these coordinates. It enforces the squared equivalent of the norm
bound, then sets `x_{i+1}=F(y*)`. There is no ball on `y` itself and no extra
post-solve ball projection. The optional inherited monotone effort constraint
is also imposed inside this NLP:

\[
\|F(y)-r_i\|_2^2\leq
  (\Delta t_i\,\beta\|u_i^{\mathrm{last\ admitted}}\|_2)^2,
\qquad \beta=0.99.
\]

The guidance control and outer objective are

\[
u_i=\frac{x_{i+1}-r_i}{\Delta t_i},\qquad
E=\frac12\sum_i\|u_i\|_2^2\Delta t_i,
\]
\[
J=C(x_N)+w_E E+w_{\rm contact}N_{\rm contact},
\qquad w_E=0.3,\quad w_{\rm contact}=1.
\]

These weights are for outer ranking. The inner objective remains the original
HardFlow surrogate with coefficient one. `C` is evaluated on normalized
planning coordinates, whereas reported trajectory length is
`sum_k ||p_{k+1}-p_k||_2` in physical coordinates. Simulator contacts are evaluated
after generation and are not differentiable NLP constraints.

## Outer iteration

1. Generate Original Flow and HardFlow with the identical sampled initial-noise
   tensor. Save the HardFlow chain as the initial archive.
2. Fit a minimum enclosing Euclidean ball at each archived tail level, including
   the entry level (`x_5` for ten steps and a five-step tail). Expand each radius
   by a fixed epsilon derived from baseline step displacements, multiplied by 16.
3. Propose an initial noise. In rounds 1–150, choose the incumbent neighborhood
   with probability 0.7, otherwise the global initial-noise ball. Candidate
   directions/radii are stratified and screened by their exact uncontrolled
   prefix; far-away candidates are preferred. This is not a uniform distribution
   over finally selected noises. The pool size is 1024; only a subset receives
   the more expensive prefix evaluation.
4. Compute and freeze that candidate's uncontrolled prefix. Reject an entry
   outside the known entry ball. Solve the tail surrogate sequentially. Prefixes
   are reused within a candidate, not across different initial noises.
5. Evaluate complete candidates, including a PD rollout for contact cost. Only
   explicit-feasible strict improvements update the archive, fitted balls, best
   noise and optional effort cap. Incomplete candidates are logged as failures.
6. After ten consecutive unsuccessful attempts, grow the global initial-noise
   radius by 1.5, capped at 30. Epsilon itself is not automatically increased.

From round 151, propose `z' = z_chain + sigma * eta`, `eta ~ N(0,I)`. For a
symmetric chain proposal the implemented log acceptance ratio is

\[
\log\alpha=\min\left(0,
-\frac{J(z')-J(z_{\rm chain})}{T}
-\frac{\|z'\|_2^2-\|z_{\rm chain}\|_2^2}{2}\right).
\]

The second term is the standard Gaussian prior on free initial-noise
coordinates. Invalid candidates are rejected. The sampler can accept an uphill
chain move while retaining the historical best noise independently.

This is **adaptive MH-style optimization**, not a guarantee of sampling a fixed
stationary distribution: archive-dependent balls, warm starts and optional
control bounds change, and chain costs are cached from their previous
evaluation. This limitation is inherited from the existing experiment.

