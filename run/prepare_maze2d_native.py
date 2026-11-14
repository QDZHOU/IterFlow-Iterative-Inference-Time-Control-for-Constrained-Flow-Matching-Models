"""Fit the original linear dynamics using compact, unpadded D4RL episodes."""
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from hardflow.config.flow_matching import FlowMatchingEvaluationConfig
from hardflow.native_maze2d_data import load_compact_data

cfg = FlowMatchingEvaluationConfig()
_, fields, normalizer = load_compact_data(cfg)
observations = [normalizer.normalize(x, 'observations') for x in fields['observations']]
actions = [normalizer.normalize(x, 'actions') for x in fields['actions']]
X = np.concatenate([x[:-1] for x in observations])
y = np.concatenate([x[1:] for x in observations])
u = np.concatenate([x[:-1] for x in actions])
model = LinearRegression().fit(np.hstack([X, u]), y)
pred = model.predict(np.hstack([X, u]))
metrics = {'overall_mse': float(mean_squared_error(y, pred)), 'overall_r2': float(r2_score(y, pred))}
path = Path('logs') / cfg.env / 'dynamics' / 'linear_model.npz'
path.parent.mkdir(parents=True, exist_ok=True)
np.savez(path, A=model.coef_[:, :4], B=model.coef_[:, 4:], c=model.intercept_, normalizer=normalizer, metrics=metrics)
print('Dynamics saved:', path, metrics)
