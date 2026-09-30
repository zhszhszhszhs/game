"""Reproducibility, output files and device selection."""
import csv
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed

ROOT = Path(__file__).resolve().parents[1]


def setup(seed, device='auto', threads=4):
    torch.set_num_threads(threads)
    set_random_seed(seed)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use --device cpu or repair the NVIDIA driver/PyTorch installation.')
    return device


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('No episodes were collected')
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(rows):
    rewards = np.asarray([r['reward'] for r in rows], dtype=float)
    return dict(episodes=len(rows), mean_reward=float(rewards.mean()),
                std_reward=float(rewards.std()), median_reward=float(np.median(rewards)),
                max_reward=float(rewards.max()), min_reward=float(rewards.min()),
                mean_episode_length=float(np.mean([r['length'] for r in rows])))


def versions():
    return dict(python=platform.python_version(),
                packages={p: version(p) for p in ['torch', 'gymnasium', 'ale-py',
                          'stable-baselines3', 'numpy', 'matplotlib', 'tensorboard']},
                cuda_available=torch.cuda.is_available(),
                cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
