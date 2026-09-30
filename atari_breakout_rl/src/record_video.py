"""Stream RGB frames to MP4 without keeping whole episodes in memory."""
import argparse
from pathlib import Path

import imageio.v2 as imageio
from stable_baselines3 import DQN
from env import ENV_SPEC, make_env
from utils import ROOT, setup, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, help='Omit for random agent')
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--seed', type=int, default=20000)
    p.add_argument('--device', default='auto')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results/videos')
    p.add_argument('--name', help='e.g. random_agent, dqn_early, dqn_final')
    args = p.parse_args()
    if args.episodes < 1:
        p.error('episodes must be positive')
    device = setup(args.seed, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or ('dqn_final' if args.model else 'random_agent')
    env = make_env(args.seed, render_mode='rgb_array')
    records = []
    try:
        model = DQN.load(args.model, env=env, device=device) if args.model else None
        for episode in range(args.episodes):
            seed = args.seed + episode
            env.seed(seed)
            env.action_space.seed(seed)
            obs = env.reset()
            path = args.output_dir / f'{name}_{episode+1:02d}.mp4'
            # One frame per agent step: original 60 Hz / action repeat 4 = 15 fps.
            with imageio.get_writer(str(path), fps=15, codec='libx264',
                                    macro_block_size=2, pixelformat='yuv420p') as writer:
                frame = env.get_images()[0]
                writer.append_data(frame)
                imageio.imwrite(args.output_dir / f'{name}_{episode+1:02d}.png', frame)
                total, length = 0.0, 0
                while True:
                    action = [env.action_space.sample()] if model is None else model.predict(obs, deterministic=True)[0]
                    obs, rewards, dones, _ = env.step(action)
                    total += float(rewards[0])
                    length += 1
                    # VecEnv auto-resets on done: do not record the next game's reset frame.
                    if dones[0]:
                        break
                    writer.append_data(env.get_images()[0])
            records.append(dict(video=str(path), seed=seed, reward=total, length=length))
            print(records[-1], flush=True)
        write_json(args.output_dir / f'{name}_metadata.json', dict(
            model=str(args.model) if args.model else None, environment=ENV_SPEC,
            fps=15, episodes=records))
    finally:
        env.close()


if __name__ == '__main__':
    main()
