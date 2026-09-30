"""Independent, seeded full-game evaluation; no parameter updates."""
import argparse
import json
from pathlib import Path

from stable_baselines3 import DQN
from env import ENV_SPEC, make_env
from utils import ROOT, setup, summary, versions, write_csv, write_json


def run_episodes(model=None, episodes=50, seed=10000, output=None):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    env = make_env(seed)
    rows = []
    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            # Reset both ALE RNG and action RNG for independently reproducible episodes.
            env.seed(episode_seed)
            env.action_space.seed(episode_seed)
            obs = env.reset()
            reward_sum, length = 0.0, 0
            while True:
                action = [env.action_space.sample()] if model is None else model.predict(obs, deterministic=True)[0]
                obs, rewards, dones, infos = env.step(action)
                reward_sum += float(rewards[0])
                length += 1
                if dones[0]:
                    assert abs(reward_sum - infos[0]['episode']['r']) < 1e-4
                    rows.append(dict(episode=episode + 1, seed=episode_seed,
                                     reward=reward_sum, length=length))
                    print(f"Episode {episode+1}/{episodes}: reward={reward_sum:g}, length={length}", flush=True)
                    break
    finally:
        env.close()
    if output:
        write_csv(output, rows)
        write_json(Path(output).with_suffix('.json'),
                   dict(summary=summary(rows), environment=ENV_SPEC, seed=seed,
                        deterministic=model is not None, versions=versions()))
    print(json.dumps(summary(rows), indent=2))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--seed', type=int, default=10000)
    p.add_argument('--device', default='auto')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results/evaluation')
    p.add_argument('--compare-random', action='store_true')
    args = p.parse_args()
    device = setup(args.seed, args.device)
    # Pass the shared environment on load to validate observation/action spaces.
    env = make_env(args.seed)
    try:
        model = DQN.load(args.model, env=env, device=device)
        dqn = run_episodes(model, args.episodes, args.seed, args.output_dir / 'dqn_results.csv')
        meta_path = args.output_dir / 'dqn_results.json'
        meta = json.loads(meta_path.read_text())
        meta.update(model=str(args.model.resolve()), training_timesteps=model.num_timesteps)
        write_json(meta_path, meta)
        if args.compare_random:
            random = run_episodes(None, args.episodes, args.seed, args.output_dir / 'random_results.csv')
            comparison = [dict(agent=name, **summary(rows)) for name, rows in [('Random', random), ('DQN', dqn)]]
            write_csv(args.output_dir / 'comparison.csv', comparison)
            print('\nAgent    Average reward    Maximum reward    Mean episode length')
            for row in comparison:
                print(f"{row['agent']:8} {row['mean_reward']:14.2f} {row['max_reward']:17.2f} {row['mean_episode_length']:22.2f}")
    finally:
        env.close()


if __name__ == '__main__':
    main()
