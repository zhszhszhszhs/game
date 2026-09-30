"""Random action baseline with the same preprocessing as DQN."""
import argparse
from pathlib import Path
from evaluate import run_episodes
from utils import ROOT, setup

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--seed', type=int, default=10000)
    p.add_argument('--output', type=Path, default=ROOT / 'results/evaluation/random_results.csv')
    args = p.parse_args()
    setup(args.seed, 'cpu')
    run_episodes(episodes=args.episodes, seed=args.seed, output=args.output)
