"""Compact evaluation data for the native maze2d backend.
Use the same episode segmentation and float32 conversion as GoalDataset,
without allocating its padded training replay buffer.
"""
import numpy as np
from .datasets.d4rl import load_environment
from .datasets.preprocessing import get_preprocess_fn
from .datasets.normalization import DatasetNormalizer


def load_compact_data(cfg):
    env = load_environment(cfg.env)
    env.seed(cfg.seed)
    raw = get_preprocess_fn(cfg.preprocess_fns, env)(env.get_dataset())
    boundaries = np.flatnonzero(raw['timeouts'] | raw['terminals'])
    if len(boundaries) > cfg.max_n_episodes:
        raise ValueError('Dataset exceeds max_n_episodes')
    starts = np.r_[0, boundaries[:-1] + 1]
    fields = {'observations': [], 'actions': [], 'next_observations': []}
    lengths = []
    for start, end in zip(starts, boundaries):
        length = end - start  # process_maze2d_episode drops the final row.
        if length > cfg.max_path_length or length <= 0:
            raise ValueError(f'Invalid episode length: {length}')
        lengths.append(length)
        for key in ('observations', 'actions'):
            fields[key].append(raw[key][start:end].astype(np.float32))
        fields['next_observations'].append(raw['observations'][start+1:end+1].astype(np.float32))
    normalizer = DatasetNormalizer(fields, cfg.normalizer, lengths)
    print(f'Compact dataset: {len(lengths)} episodes, {sum(lengths)} steps')
    return env, fields, normalizer
