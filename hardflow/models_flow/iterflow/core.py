"""Shared normalized-coordinate sampler, cost and specification checks."""
from dataclasses import dataclass
import hashlib
import json
import time
import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def array_digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(a.dtype.str.encode()); h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Cost:
    kind: str = "hf"
    coefficient: float = 1.0

    def __post_init__(self):
        if self.kind not in ("hf", "oc") or not np.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("Cost must be hf or oc with a finite nonnegative coefficient")

    def stage(self, reference, target, dt):
        scale = self.coefficient / 2 if self.kind == "hf" else self.coefficient / (2 * dt)
        return scale * np.sum((np.asarray(target) - reference) ** 2, axis=-1)


def bounds(value, n):
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        a = np.full(n, float(a))
    if a.shape != (n,) or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError(f"u_bar must be scalar or {n} nonnegative finite values")
    return a


def validate_spec(spec):
    grid = np.asarray(spec["time_grid"], dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("Time grid must be finite and strictly increasing")
    if not isinstance(spec["free_dim"], int) or spec["free_dim"] < 1:
        raise ValueError("Invalid free coordinate dimension")
    bounds(spec["u_bar"], len(grid)-1)
    Cost(**spec["cost"])
    for key in ("reach", "tie", "feasibility"):
        value = spec["tolerances"][key]
        if not np.isfinite(value) or value < 0: raise ValueError("Invalid numerical tolerance")


def sample(backend, noise, spec):
    validate_spec(spec)
    n = len(spec["time_grid"]) - 1; dt = np.diff(spec["time_grid"])
    cost = Cost(**spec["cost"]); ub = bounds(spec["u_bar"], n)
    X = [backend.free_noise(noise)]; U = []; refs = []; chosen = []; stages = []; errors = []
    timers = dict(flow=0.0, retrieval=0.0, ipopt=0.0, total=0.0)
    start = time.perf_counter(); fallback = 0
    for i in range(n):
        before = time.perf_counter(); reference = backend.natural(i, X[-1])
        timers["flow"] += time.perf_counter() - before
        before = time.perf_counter()
        j = -1
        timers["retrieval"] += time.perf_counter() - before
        fallback += 1
        next_x, detail = backend.fallback(i, X[-1], reference)
        timers["flow"] += detail.get("flow_seconds", 0.0)
        timers["ipopt"] += detail.get("ipopt_seconds", 0.0)
        if detail.get("error"): errors.append({"layer": i, "reason": detail["error"]})
        if not np.isfinite(next_x).all(): raise ValueError(f"Nonfinite generated state at layer {i+1}")
        u = (next_x - reference) / dt[i]
        if j >= 0 and np.linalg.norm(u) > ub[i] + spec["tolerances"]["reach"] / dt[i]:
            raise AssertionError("Memory control exceeds reachability tolerance")
        stages.append(float(cost.stage(reference, next_x, dt[i])))
        refs.append(reference); X.append(next_x); U.append(u); chosen.append(j)
    feasible, terminal_cost, residual = backend.terminal(X[-1])
    norms = np.linalg.norm(U, axis=1)
    selected = [j for j in chosen if j >= 0]
    switches = sum(a != b for a, b in zip(selected, selected[1:]))
    terminal_sample = chosen[-1] if chosen[-1] >= 0 else None
    timers["total"] = time.perf_counter() - start
    report = dict(terminal_feasible=bool(feasible), residual=residual, terminal_cost=float(terminal_cost),
                  stage_cost=float(sum(stages)), total_cost=float(terminal_cost + sum(stages)),
                  memory_choices=chosen, memory_hits=len(selected), fallback_count=fallback,
                  first_memory_layer=next((i+1 for i,j in enumerate(chosen) if j>=0), None),
                  record_switches=switches, terminal_sample_index=terminal_sample,
                  terminal_sample_seed=None,
                  control_norms=norms.tolist(), control_bound_violations=int(np.sum(norms > ub + spec["tolerances"]["reach"]/dt)),
                  suffix_certified=bool(terminal_sample is not None and feasible), errors=errors, timings=timers)
    return {"X": np.array(X), "U": np.array(U), "refs": np.array(refs), "report": report}
