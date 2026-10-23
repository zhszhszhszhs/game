"""Independent, seeded full-game evaluation; no parameter updates."""
import argparse
import json
from pathlib import Path

from stable_baselines3.common.utils import check_for_correct_spaces
from model_io import algorithm_name, load_model
from env import environment_spec, make_env
from utils import ROOT, setup, summary, versions, write_csv, write_json


def run_episodes(model=None, episodes=50, seed=10000, output=None, on_episode=None):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    spec = environment_spec(model)
    env = make_env(seed)
    rows = []
    try:
        if model is not None:
            check_for_correct_spaces(env, model.observation_space, model.action_space)
        for episode in range(episodes):
            episode_seed = seed + episode
            # Reset both game RNG and action RNG for independently reproducible episodes.
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
                                     reward=reward_sum, length=length,
                                     truncated=bool(infos[0].get('TimeLimit.truncated', False)),
                                     remaining_lives=infos[0].get('lives')))
                    print(f"Episode {episode+1}/{episodes}: reward={reward_sum:g}, length={length}", flush=True)
                    if on_episode is not None:
                        on_episode(rows[-1], len(rows))
                    break
    finally:
        env.close()
    if output:
        write_csv(output, rows)
        write_json(Path(output).with_suffix('.json'),
                   dict(summary=summary(rows), environment=spec, seed=seed,
                        deterministic=model is not None, versions=versions()))
    print(json.dumps(summary(rows), indent=2))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--seed', type=int, default=10000)
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results/evaluation')
    p.add_argument('--compare-random', action='store_true')
    args = p.parse_args()
    if args.threads < 1:
        p.error('--threads must be positive')
    device = setup(args.seed, args.device, args.threads)
    model = load_model(args.model, device=device)
    algorithm = algorithm_name(model)
    trained = run_episodes(model, args.episodes, args.seed, args.output_dir / f'{algorithm}_results.csv')
    meta_path = args.output_dir / f'{algorithm}_results.json'
    meta = json.loads(meta_path.read_text())
    meta.update(algorithm=algorithm, model=str(args.model.resolve()), training_timesteps=model.num_timesteps,
                training_environment=environment_spec(model),
                training_reward=getattr(model, 'breakout_training_reward', None))
    write_json(meta_path, meta)
    if args.compare_random:
        random = run_episodes(None, args.episodes, args.seed, args.output_dir / 'random_results.csv')
        comparison = [dict(agent=name, **summary(rows)) for name, rows in [('Random', random), (algorithm.upper(), trained)]]
        write_csv(args.output_dir / 'comparison.csv', comparison)
        print('\nAgent    Average reward    Maximum reward    Mean episode length')
        for row in comparison:
            print(f"{row['agent']:8} {row['mean_reward']:14.2f} {row['max_reward']:17.2f} {row['mean_episode_length']:22.2f}")


if __name__ == '__main__':
    main()
