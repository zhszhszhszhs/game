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
from warm_start_dqn import WarmStartDQN as DQN
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.utils import LinearSchedule
from callbacks import HealthCallback, TrackedEvalCallback, RunState
from stable_baselines3.common.logger import configure
from env import environment_spec, make_env
from utils import ROOT, setup, versions, write_json
from replay_io import load_replay, save_replay


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=Path, default=ROOT / 'configs/dqn_custom.yaml')
    known, _ = config_parser.parse_known_args()
    p = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    defaults = yaml.safe_load(known.config.read_text())
    for name in ['total_timesteps', 'seed', 'buffer_size', 'learning_starts', 'batch_size',
                 'train_freq', 'gradient_steps', 'target_update_interval', 'checkpoint_freq',
                 'eval_freq', 'n_eval_episodes', 'threads']:
        p.add_argument('--' + name.replace('_', '-'), type=int, default=defaults[name])
    for name in ['learning_rate', 'gamma', 'exploration_fraction', 'exploration_final_eps']:
        p.add_argument('--' + name.replace('_', '-'), type=float, default=defaults[name])
    p.add_argument('--exploration-initial-eps', type=float,
                   default=defaults.get('exploration_initial_eps', 1.0),
                   help='Initial epsilon for this stage; use a low value when fine-tuning a trained model')
    p.add_argument('--eval-seed', type=int, default=defaults.get('eval_seed'))
    p.add_argument('--final-learning-rate', type=float, default=defaults.get('final_learning_rate'),
                   help='Optional linear decay from learning-rate over this stage')
    p.add_argument('--early-stop-patience', type=int, default=defaults.get('early_stop_patience', 0),
                   help='Stop after this many consecutive evaluations without a new best mean; 0 disables')
    p.add_argument('--n-steps', type=int, default=defaults.get('n_steps', 1),
                   help='SB3 n-step Bellman return horizon; changing this requires fresh replay')
    p.add_argument('--compress-replay-buffer', action=argparse.BooleanOptionalAction,
                   default=defaults.get('compress_replay_buffer', False))
    p.add_argument('--evaluate-at-start', action=argparse.BooleanOptionalAction,
                   default=defaults.get('evaluate_at_start', False))
    p.add_argument('--device', default=defaults['device'])
    p.add_argument('--log-dir', type=Path, default=ROOT / 'logs')
    p.add_argument('--model-dir', type=Path, default=ROOT / 'models')
    p.add_argument('--run-name', default=None)
    p.add_argument('--save-replay-buffer', action=argparse.BooleanOptionalAction,
                   default=defaults.get('save_replay_buffer', False),
                   help='Save replay on final/interrupted exit, not at each periodic checkpoint')
    p.add_argument('--save-best-replay', action=argparse.BooleanOptionalAction,
                   default=defaults.get('save_best_replay', False),
                   help='Also save matching replay whenever the best mean model improves')
    p.add_argument('--reset-timesteps', action=argparse.BooleanOptionalAction,
                   default=defaults.get('reset_timesteps', False),
                   help='On resume, start a new stage counter and epsilon schedule; keep weights and optimizer')
    p.add_argument('--resume', type=Path, help='Model .zip; total-timesteps means ADDITIONAL steps')
    p.add_argument('--replay-buffer', type=Path, help='Replay .pkl matching the resumed model')
    p.add_argument('--policy-warmup', action=argparse.BooleanOptionalAction,
                   default=defaults.get('policy_warmup', False),
                   help='Fill fresh replay using resumed epsilon-greedy policy, while delaying gradient updates')
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
    if args.policy_warmup and (not args.resume or args.replay_buffer):
        p.error('--policy-warmup requires --resume and fresh replay (omit --replay-buffer)')
    if not 0 <= args.exploration_final_eps <= args.exploration_initial_eps <= 1:
        p.error('Exploration must satisfy 0 <= final-eps <= initial-eps <= 1')
    if not 0 < args.exploration_fraction <= 1:
        p.error('exploration-fraction must be in (0, 1]')
    if args.n_steps < 1 or args.buffer_size <= args.n_steps:
        p.error('n-steps must be positive and smaller than buffer-size')
    if not args.replay_buffer and args.learning_starts < args.n_steps:
        p.error('Fresh replay requires learning-starts >= n-steps')
    if not args.replay_buffer and args.total_timesteps <= args.learning_starts:
        p.error('total-timesteps must exceed fresh replay warm-up')
    if args.learning_rate <= 0 or (args.final_learning_rate is not None and
                                  not 0 < args.final_learning_rate <= args.learning_rate):
        p.error('Learning rates must satisfy 0 < final-learning-rate <= learning-rate')
    if args.final_learning_rate is not None and args.resume and not args.reset_timesteps:
        p.error('Stage learning-rate decay on resume requires --reset-timesteps')
    if args.early_stop_patience < 0:
        p.error('early-stop-patience must be nonnegative')
    if args.early_stop_patience and not args.evaluate_at_start:
        p.error('Early stopping requires --evaluate-at-start to protect the starting policy')
    return args


def main():
    args = parse_args()
    device = setup(args.seed, args.device, args.threads)
    name = args.run_name or datetime.now().strftime('dqn_%Y%m%d_%H%M%S')
    log_dir, model_dir = args.log_dir / name, args.model_dir / name
    # Avoid silently overwriting an earlier experiment.
    log_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    spec = environment_spec()
    metadata = dict(arguments=vars(args), device=device, versions=versions(), environment=spec)
    write_json(log_dir / 'config.json', metadata)
    env = make_env(args.seed, log_dir / 'train.monitor.csv')
    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed + 1000
    eval_env = make_env(eval_seed, log_dir / 'eval/eval.monitor.csv')
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
                    'exploration_fraction', 'exploration_initial_eps', 'exploration_final_eps', 'n_steps']}
        if args.final_learning_rate is not None:
            kwargs['learning_rate'] = LinearSchedule(args.learning_rate, args.final_learning_rate, 1.0)
        if args.resume:
            model = DQN.load(args.resume, env=env, device=device, seed=args.seed,
                             tensorboard_log=str(log_dir), replay_buffer_class=None,
                             replay_buffer_kwargs={}, **kwargs)
            source_spec = environment_spec(model)
            metadata['source_environment'] = source_spec
            model.breakout_env_spec = spec
            if args.replay_buffer:
                load_replay(model, args.replay_buffer)
            else:
                # A .zip never contains replay. Warm up again before gradient updates.
                model.learning_starts = (0 if args.reset_timesteps else model.num_timesteps) + args.learning_starts
                print('Resume without replay: fresh buffer and another warm-up; not an exact continuation.')
        else:
            model = DQN('CnnPolicy', env, device=device, seed=args.seed, verbose=1,
                        tensorboard_log=str(log_dir), **kwargs)
        model.breakout_env_spec = spec  # Saved inside every checkpoint, including the initial baseline.
        model.policy_warmup = args.policy_warmup
        if args.reset_timesteps or not args.resume:
            # DQN.load restores the old exploration_rate. Set the first action's
            # epsilon too; subsequent values come from SB3's LinearSchedule.
            model.exploration_rate = args.exploration_initial_eps
        metadata['source_timesteps'] = model.num_timesteps
        metadata['replay_buffer_class'] = type(model.replay_buffer).__name__
        metadata['start_timesteps'] = 0 if args.reset_timesteps else model.num_timesteps
        start_updates = model._n_updates
        state.target = metadata['start_timesteps'] + args.total_timesteps
        write_json(log_dir / 'config.json', metadata)
        model.set_logger(configure(str(log_dir), ['stdout', 'csv', 'tensorboard']))
        print('Device:', model.device, 'Observation:', env.observation_space, flush=True)
        print('Bellman return steps:', model.n_steps,
              'Replay:', type(model.replay_buffer).__name__, flush=True)
        print('Fresh replay warm-up actions:',
              'resumed epsilon-greedy policy' if args.policy_warmup else 'uniform random', flush=True)
        # SB3 stops when its counter EXCEEDS max_no_improvement_evals.
        # The initial evaluation establishes the reference, then count exactly patience checks.
        stopper = (StopTrainingOnNoModelImprovement(args.early_stop_patience - 1,
                   min_evals=1, verbose=1) if args.early_stop_patience else None)
        callbacks = [CheckpointCallback(save_freq=args.checkpoint_freq,
                        save_path=str(model_dir / 'checkpoints'), name_prefix='dqn_breakout',
                        save_replay_buffer=False),
                     TrackedEvalCallback(eval_env, state=state, eval_seed=eval_seed,
                        callback_after_eval=stopper,
                        evaluate_at_start=args.evaluate_at_start,
                        eval_start_step=model.learning_starts + 1 if args.policy_warmup else 0,
                        save_best_replay=args.save_best_replay,
                        compress_replay=args.compress_replay_buffer,
                        stable_model_save_path=model_dir / 'stable_model',
                        best_model_save_path=str(model_dir / 'best_model'),
                        log_path=str(log_dir / 'eval'), eval_freq=args.eval_freq,
                        n_eval_episodes=args.n_eval_episodes, deterministic=True), HealthCallback(state)]
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks,
                    reset_num_timesteps=(not bool(args.resume) or args.reset_timesteps), log_interval=10)
        stopped_early = bool(stopper and
                             stopper.no_improvement_evals >= args.early_stop_patience)
        status = 'early_stopped' if stopped_early else 'completed'
        stop_reason = 'no_development_mean_improvement' if stopped_early else 'step_budget_reached'
        model.save(model_dir / 'final_model')
        replay_path = None
        if args.save_replay_buffer:
            state.update(model, 'saving_replay')
            replay_path = save_replay(model, model_dir / 'final_replay_buffer.pkl',
                                      args.compress_replay_buffer)
        write_json(log_dir / 'training_summary.json', dict(
            status=status, stop_reason=stop_reason, target_timesteps=state.target,
            timesteps=model.num_timesteps, gradient_updates=model._n_updates,
            stage_gradient_updates=model._n_updates-start_updates,
            replay_size=model.replay_buffer.size(), elapsed_seconds=time.monotonic()-started,
            evaluation_seconds=state.eval_seconds, source_timesteps=metadata['source_timesteps'],
            n_steps=model.n_steps, replay_buffer_path=replay_path,
            environment=spec,
            device=str(model.device), final_model=str(model_dir / 'final_model.zip')))
        state.update(model, status, stop_reason=stop_reason)
    except KeyboardInterrupt:
        state.update(model, 'saving_interrupted')
        if model is not None:
            model.save(model_dir / 'interrupted_model')
            if args.save_replay_buffer:
                save_replay(model, model_dir / 'interrupted_replay_buffer.pkl',
                            args.compress_replay_buffer)
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
