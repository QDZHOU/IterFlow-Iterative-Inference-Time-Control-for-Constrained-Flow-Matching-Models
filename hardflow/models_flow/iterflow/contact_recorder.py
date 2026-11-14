"""Record particle-wall contacts during native MuJoCo execution."""
import numpy as np

class ContactRecorder:
    """Observe the real env.step while retaining the existing PD executor."""
    def __init__(self, env):
        self.env = env
        self.contacts = []
        self.penetrations = []
        self.velocity_clip_steps = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        self.velocity_clip_steps.append(bool(np.any(np.abs(self.env.data.qvel) > 5.)))
        result = self.env.step(action)
        pairs = []
        penetration = 0.
        for contact in self.env.data.contact[:self.env.data.ncon]:
            names = [self.env.model.geom(int(g)).name or '' for g in (contact.geom1, contact.geom2)]
            if 'particle_geom' in names and any(n.startswith('wall_') for n in names):
                pairs.append(names)
                penetration = max(penetration, -float(contact.dist))
        self.contacts.append(pairs)
        self.penetrations.append(max(penetration, 0.))
        return result
