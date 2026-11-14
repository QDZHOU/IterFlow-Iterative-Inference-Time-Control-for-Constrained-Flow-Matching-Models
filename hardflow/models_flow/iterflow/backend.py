"""Shared coordinates and unchanged HardFlow Algorithm 1 affine step."""
import hashlib
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()


class Coordinates:
    def __init__(self, length, action_dim, state_dim, conditions):
        self.shape = (length, action_dim + state_dim)
        self.template = np.zeros(self.shape, dtype=np.float32)
        self.mask = np.ones(self.shape, dtype=bool)
        for layer, value in conditions.items():
            if not 0 <= int(layer) < length or np.asarray(value).shape != (state_dim,):
                raise ValueError("Invalid condition frame or state dimension")
            self.template[int(layer), action_dim:] = value
            self.mask[int(layer), action_dim:] = False
        self.free_dim = int(self.mask.sum())

    def free(self, full):
        full = np.asarray(full)
        if full.shape == (1, *self.shape): full = full[0]
        if full.shape != self.shape: raise ValueError("Full planning tensor shape mismatch")
        return full[self.mask].copy()

    def full(self, x):
        if np.asarray(x).shape != (self.free_dim,): raise ValueError("Free coordinate shape mismatch")
        full = self.template.astype(np.asarray(x).dtype, copy=True)
        full[self.mask] = x
        return full

class HardFlowBackend:
    def spec(self):
        spec = dict(self.identity)
        spec.update(cost=self.settings["cost"], u_bar=self.settings["u_bar"], tolerances=self.settings["tolerances"])
        return spec

    def noise(self, seed):
        generator = self.torch.Generator(device="cpu").manual_seed(seed)
        return self.torch.randn(1, *self.coordinates.shape, generator=generator).numpy()

    def free_noise(self, noise):
        return self.coordinates.free(noise)

    def natural(self, i, x):
        return x + self.velocity(x, float(self.grid[i])) * (self.grid[i+1]-self.grid[i])

    def fallback(self, i, x, reference):
        # Per-layer extraction of upstream hardflow_new_forward, preserving the
        # t_next multiplier, terminal prediction, solver/debug fallback, and late flag.
        policy = self.policy; next_t = float(self.grid[i+1]); detail = {}
        if self.cfg.hardflow_activation == "late" and i < getattr(self,"control_start",self.cfg.ode_t_steps//2):
            return reference.copy(), detail
        if self.cfg.hardflow_activation not in ("all", "late"): raise ValueError("Invalid activation")
        start = time.perf_counter()
        predicted = reference + (1.0-next_t)*self.velocity(reference,next_t)
        detail["flow_seconds"] = time.perf_counter()-start
        opti = policy.oc_cs_opti
        opti.set_value(policy.oc_t_param,next_t)
        opti.set_value(policy.oc_X_terminal_predicted_ref,predicted)
        opti.set_initial(policy.oc_X_terminal_predicted,predicted)
        start = time.perf_counter()
        try:
            solution = opti.solve_limited()
            terminal = solution.value(policy.oc_X_terminal_predicted)
            if not solution.stats().get("success", False):
                detail["error"] = "IPOPT returned " + solution.stats().get("return_status", "unknown status")
        except RuntimeError as error:
            detail["error"] = str(error)
            terminal = opti.debug.value(policy.oc_X_terminal_predicted)
        detail["ipopt_seconds"] = time.perf_counter()-start
        return np.asarray(reference + next_t*(terminal-predicted)).reshape(-1), detail

    def native_reference(self, seed):
        """Call the unmodified upstream sampler for the empty-memory equivalence test."""
        with self.torch.random.fork_rng():
            self.torch.manual_seed(seed)
            self.policy(self.conditions, batch_size=1)
        return np.stack([self.coordinates.free(full) for full in self.policy.captured_chain[0]])
