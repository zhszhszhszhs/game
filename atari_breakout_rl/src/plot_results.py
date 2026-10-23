"""Report-ready plots from raw Monitor CSV and independent evaluation CSV."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import load_results
from utils import ROOT


def save(path, title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25, axis='y')
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log-dir', type=Path, required=True, help='One training run directory')
    p.add_argument('--evaluation-dir', type=Path, default=ROOT / 'results/evaluation')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results/figures')
    p.add_argument('--window', type=int, default=100)
    args = p.parse_args()
    if args.window < 1:
        p.error('window must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_results(str(args.log_dir))
    if data.empty:
        raise ValueError('No complete training episodes in Monitor log')
    steps = data.l.cumsum()
    algorithm = 'dqn'
    config = args.log_dir / 'config.json'
    if config.exists():
        import json
        metadata = json.loads(config.read_text())
        offset = metadata.get('start_timesteps', 0)
        algorithm = metadata.get('algorithm', 'dqn')
        steps += offset
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, data.r, alpha=0.7, linewidth=0.8)
    save(args.output_dir / 'reward_vs_steps.png', f'{algorithm.upper()} Training Episode Reward',
         'Training steps (agent decisions)', 'Native episode reward')
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, data.r, alpha=0.2, label='Episode reward')
    plt.plot(steps, data.r.rolling(args.window, min_periods=1).mean(),
             label=f'Rolling mean ({args.window} episodes)')
    plt.legend()
    save(args.output_dir / 'smoothed_reward.png', f'{algorithm.upper()} Smoothed Training Reward',
         'Training steps (agent decisions)', 'Native episode reward')
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, data.l, alpha=0.7, linewidth=0.8)
    save(args.output_dir / 'episode_length.png', f'{algorithm.upper()} Training Episode Length',
         'Training steps (agent decisions)', 'Episode length (agent decisions)')
    evaluations = args.log_dir / 'eval/evaluations.npz'
    if evaluations.exists():
        with np.load(evaluations) as data:
            eval_steps, rewards = data['timesteps'], data['results']
        plt.figure(figsize=(8, 4.5))
        plt.plot(eval_steps, rewards.mean(axis=1), marker='o', label='Mean')
        plt.plot(eval_steps, np.median(rewards, axis=1), marker='.', label='Median')
        plt.legend()
        save(args.output_dir / 'development_evaluation.png', f'{algorithm.upper()} Development Evaluation',
             'Training steps (agent decisions)', 'Native episode reward')
    random = pd.read_csv(args.evaluation_dir / 'random_results.csv')
    trained = pd.read_csv(args.evaluation_dir / f'{algorithm}_results.csv')
    if len(random) != len(trained) or not np.array_equal(random.seed, trained.seed):
        raise ValueError('Comparison requires equal episode counts and identical episode seeds')
    means = [random.reward.mean(), trained.reward.mean()]
    stds = [random.reward.std(ddof=0), trained.reward.std(ddof=0)]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(['Random', algorithm.upper()], means, yerr=stds, capsize=6, color=['#8995a7', '#2676b8'])
    plt.bar_label(bars, fmt='%.2f', padding=4)
    save(args.output_dir / f'{algorithm}_vs_random.png', f'Independent Evaluation (n={len(trained)}, error bars: SD)',
         'Agent', 'Mean native episode reward')
    print('SD describes episode variability, not a confidence interval; one training seed is not algorithm-level evidence.')


if __name__ == '__main__':
    main()
