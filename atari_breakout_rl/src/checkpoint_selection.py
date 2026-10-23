"""Choose between mean-best and median-best without using final test scores."""
import json
from pathlib import Path


def choose_checkpoint(model_dir, mode='mean', mean_tolerance=.05):
    if mode not in ('mean', 'stable_near_mean') or not 0 <= mean_tolerance < 1:
        raise ValueError('Invalid checkpoint selection rule')
    root = Path(model_dir)
    mean_path = root / 'best_model/best_model.zip'
    if not mean_path.exists():
        return root / 'final_model.zip', dict(criterion='final_model_fallback')
    if mode == 'mean':
        return mean_path, dict(criterion='mean_reward')
    mean = json.loads((mean_path.parent / 'selection.json').read_text())
    stable_path = root / 'stable_model/best_model.zip'
    stable_file = stable_path.parent / 'selection.json'
    selected, stable = mean_path, None
    floor = mean['mean_reward'] - mean_tolerance * abs(mean['mean_reward'])
    if stable_path.exists() and stable_file.exists():
        stable = json.loads(stable_file.read_text())
        if (stable['mean_reward'] >= floor and
                (stable['median_reward'], stable['mean_reward']) >
                (mean['median_reward'], mean['mean_reward'])):
            selected = stable_path
    return selected, dict(criterion=mode, mean_tolerance=mean_tolerance,
        minimum_mean_reward=floor, mean_best=mean, median_best=stable,
        selected_kind='stable_model' if selected == stable_path else 'best_model')
