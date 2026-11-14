"""Continuous geometry checks for HardFlow obstacles."""
import numpy as np

def segment_shape_closest(a, b, power):
    """Exact polynomial stationary-point search for sum_j q_j(s)^power, s in [0,1]."""
    from numpy.polynomial import Polynomial
    p = sum((Polynomial([v, d])**power for v, d in zip(a, b-a)), Polynomial([0.]))
    candidates = [0., 1.]
    for root in p.deriv().roots():
        if abs(root.imag) < 1e-9 and 0 <= root.real <= 1: candidates.append(float(root.real))
    s = min(candidates, key=lambda value: float(p(value)))
    return float(p(s)), np.asarray(a)+s*(np.asarray(b)-a)

def segment_shape_minimum(a, b, power):
    return segment_shape_closest(a, b, power)[0]

class RefinementMaze2D:
    def validate_path(self, x):
        # Same two obstacles and same margin as the actual upstream symbolic NLP.
        # Affine unnormalization without clipping; include conditioned endpoints.
        full = self.coordinates.full(x)[:, self.cfg.action_dim:]
        norm = self.normalizer.normalizers['observations']
        points = (full[:, :2]+1)/2*(norm.maxs[:2]-norm.mins[:2])+norm.mins[:2]
        violation = 0.; minima = []
        for center, power in ((np.asarray(getattr(self.cfg,'maze_circle_center',(4.5,5.3))), 2), (np.asarray(getattr(self.cfg,'maze_square_center',(1.5,4.8))), 4)):
            q = (points-center)/.8
            minimum = min(segment_shape_minimum(a, b, power) for a, b in zip(q[:-1], q[1:]))
            minima.append(minimum)
            violation = max(violation, 1+self.cfg.ellipse_margin-minimum)
        return dict(feasible=bool(np.isfinite(points).all() and violation <= self.settings['tolerances']['feasibility']),
                    max=float(violation), obstacle_minima=minima,
                    scope='all straight waypoint segments against the two upstream p=2/p=4 obstacles; not maze wall geometry or robot execution')
