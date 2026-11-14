"""Uncontrolled flow baseline and raw physical-coordinate conversion."""
import numpy as np

class NaturalFlow:
    """Use only the same natural generation update, without HardFlow corrections."""
    def __init__(self, backend): self.backend = backend
    def free_noise(self, noise): return self.backend.free_noise(noise)
    def natural(self, i, x): return self.backend.natural(i, x)
    def fallback(self, i, x, reference): return reference.copy(), {}
    def terminal(self, x): return self.backend.terminal(x)


def physical_observations(backend, x):
    full = backend.coordinates.full(x)
    observations = full[:, backend.cfg.action_dim:]
    normalizer = backend.normalizer.normalizers["observations"]
    # Preserve raw generated coordinates: upstream unnormalize may clip them.
    return (observations + 1) / 2 * (normalizer.maxs-normalizer.mins) + normalizer.mins
