"""SB3 DQN: NatureCNN, online/target Q networks, replay and epsilon-greedy.

SB3 computes a terminal-masked Bellman target and applies Huber loss,
then gradient descent. This project does not reimplement the algorithm.
"""
import argparse
from datetime import datetime
from pathlib import Path
import time
import signal

import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from callbacks import HealthCallback, TrackedEvalCallback, RunState
from stable_baselines3.common.logger import configure
from env import ENV_SPEC, make_env
from utils import ROOT, setup, versions, write_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/dqn.yaml')
    known, _ = p.parse_known_args()
    defaults = yaml.safe_load(known.config.read_text())
    for name in ['total_timesteps', 'seed', 'buffer_size', 'learning_starts', 'batch_size',
                 'train_freq', 'gradient_steps', 'target_update_interval', 'checkpoint_freq',
                 'eval_freq', 'n_eval_episodes', 'threads']:
        p.add_argument('--' + name.replace('_', '-'), type=int, default=defaults[name])
    for name in ['learning_rate', 'gamma', 'exploration_fraction', 'exploration_final_eps']:
        p.add_argument('--' + name.replace('_', '-'), type=float, default=defaults[name])
    p.add_argument('--device', default=defaults['device'])
    p.add_argument('--log-dir', type=Path, default=ROOT / 'logs')
    p.add_argument('--model-dir', type=Path, default=ROOT / 'models')
    p.add_argument('--run-name', default=None)
    p.add_argument('--save-replay-buffer', action=argparse.BooleanOptionalAction,
                   default=defaults.get('save_replay_buffer', False),
                   help='Save replay on final/interrupted exit, not at each periodic checkpoint')
    p.add_argument('--reset-timesteps', action=argparse.BooleanOptionalAction,
                   default=defaults.get('reset_timesteps', False),
                   help='On resume, start a new stage counter and epsilon schedule; keep weights and optimizer')
    p.add_argument('--resume', type=Path, help='Model .zip; total-timesteps means ADDITIONAL steps')
    p.add_argument('--replay-buffer', type=Path, help='Replay .pkl matching the resumed model')
    args = p.parse_args()
    if args.total_timesteps <= 0 or args.buffer_size <= args.batch_size:
        p.error('total-timesteps must be positive and buffer-size must exceed batch-size')
    if min(args.train_freq, args.target_update_interval, args.checkpoint_freq, args.eval_freq,
           args.n_eval_episodes, args.threads) <= 0:
        p.error('Frequencies, threads and evaluation episodes must be positive')
    if not args.resume and args.total_timesteps <= args.learning_starts:
        p.error('total-timesteps must exceed learning-starts to actually train')
    if args.replay_buffer and not args.resume:
        p.error('--replay-buffer requires --resume')
    return args


def main():
    args = parse_args()
    device = setup(args.seed, args.device, args.threads)
    name = args.run_name or datetime.now().strftime('dqn_%Y%m%d_%H%M%S')
    log_dir, model_dir = args.log_dir / name, args.model_dir / name
    # Avoid silently overwriting an earlier experiment.
    log_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    metadata = dict(arguments=vars(args), device=device, versions=versions(), environment=ENV_SPEC)
    write_json(log_dir / 'config.json', metadata)
    env = make_env(args.seed, log_dir / 'train.monitor.csv')
    eval_env = make_env(args.seed + 1000, log_dir / 'eval/eval.monitor.csv')
    model = None
    state = RunState(log_dir, args.total_timesteps)
    state.update(None, 'initializing')
    started = time.monotonic()
    start_updates = 0
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    previous_term = signal.signal(signal.SIGTERM, interrupt)
    try:
        kwargs = {k: getattr(args, k) for k in ['learning_rate', 'buffer_size', 'learning_starts',
                    'batch_size', 'gamma', 'train_freq', 'gradient_steps', 'target_update_interval',
                    'exploration_fraction', 'exploration_final_eps']}
        if args.resume:
            model = DQN.load(args.resume, env=env, device=device, seed=args.seed,
                             tensorboard_log=str(log_dir), **kwargs)
            if args.replay_buffer:
                model.load_replay_buffer(args.replay_buffer)
            else:
                # A .zip never contains replay. Warm up again before gradient updates.
                model.learning_starts = (0 if args.reset_timesteps else model.num_timesteps) + args.learning_starts
                print('Resume without replay: fresh buffer and another warm-up; not an exact continuation.')
        else:
            model = DQN('CnnPolicy', env, device=device, seed=args.seed, verbose=1,
                        tensorboard_log=str(log_dir), **kwargs)
        metadata['source_timesteps'] = model.num_timesteps
        metadata['start_timesteps'] = 0 if args.reset_timesteps else model.num_timesteps
        start_updates = model._n_updates
        state.target = metadata['start_timesteps'] + args.total_timesteps
        write_json(log_dir / 'config.json', metadata)
        model.set_logger(configure(str(log_dir), ['stdout', 'csv', 'tensorboard']))
        print('Device:', model.device, 'Observation:', env.observation_space, flush=True)
        callbacks = [CheckpointCallback(save_freq=args.checkpoint_freq,
                        save_path=str(model_dir / 'checkpoints'), name_prefix='dqn_breakout',
                        save_replay_buffer=False),
                     TrackedEvalCallback(eval_env, state=state, eval_seed=args.seed + 1000, best_model_save_path=str(model_dir / 'best_model'),
                        log_path=str(log_dir / 'eval'), eval_freq=args.eval_freq,
                        n_eval_episodes=args.n_eval_episodes, deterministic=True), HealthCallback(state)]
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks,
                    reset_num_timesteps=(not bool(args.resume) or args.reset_timesteps), log_interval=10)
        model.save(model_dir / 'final_model')
        if args.save_replay_buffer:
            model.save_replay_buffer(model_dir / 'final_replay_buffer.pkl')
        write_json(log_dir / 'training_summary.json', dict(
            status='completed', timesteps=model.num_timesteps, gradient_updates=model._n_updates,
            stage_gradient_updates=model._n_updates-start_updates,
            replay_size=model.replay_buffer.size(), elapsed_seconds=time.monotonic()-started,
            evaluation_seconds=state.eval_seconds, source_timesteps=metadata['source_timesteps'],
            device=str(model.device), final_model=str(model_dir / 'final_model.zip')))
        state.update(model, 'completed')
    except KeyboardInterrupt:
        state.update(model, 'saving_interrupted')
        if model is not None:
            model.save(model_dir / 'interrupted_model')
            if args.save_replay_buffer:
                model.save_replay_buffer(model_dir / 'interrupted_replay_buffer.pkl')
        write_json(log_dir / 'training_summary.json', dict(status='interrupted',
                    timesteps=model.num_timesteps if model else 0))
        state.update(model, 'interrupted')
        raise
    except Exception as error:
        state.update(model, 'failed', error=str(error))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        env.close()
        eval_env.close()


if __name__ == '__main__':
    main()
