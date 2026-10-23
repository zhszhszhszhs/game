"""Train SB3 PPO: CNN actor/critic, GAE and clipped policy updates."""
import argparse
from datetime import datetime
from pathlib import Path
import signal
import time
import math
import json

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import LinearSchedule
import yaml

from callbacks import HealthCallback, RunState, TrackedEvalCallback
from checkpoint_selection import choose_checkpoint
from env import environment_spec, make_env
from model_io import load_model
from ppo_tricks import EntropySchedule, split_actor_critic
from balanced_ppo import BalancedPPO, migrate_to_balanced
from dual_rate_ppo import DualRatePPO, migrate_to_dual_rate
from adaptive_kl_ppo import AdaptiveKLPPO, migrate_to_adaptive_kl, validate_controller
from utils import ROOT, setup, summary, versions, write_json


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=Path, default=ROOT / 'configs/ppo_custom.yaml')
    known, _ = config_parser.parse_known_args(argv)
    defaults = yaml.safe_load(known.config.read_text())
    p = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    for name in ['total_timesteps', 'seed', 'n_envs', 'n_steps', 'batch_size', 'n_epochs',
                 'threads', 'checkpoint_freq', 'eval_freq', 'n_eval_episodes', 'eval_seed']:
        p.add_argument('--' + name.replace('_', '-'), type=int, default=defaults[name])
    for name in ['learning_rate', 'gamma', 'gae_lambda', 'clip_range', 'ent_coef',
                 'vf_coef', 'max_grad_norm', 'target_kl']:
        p.add_argument('--' + name.replace('_', '-'), type=float, default=defaults[name])
    p.add_argument('--device', default=defaults['device'])
    p.add_argument('--run-name')
    p.add_argument('--log-dir', type=Path, default=ROOT / 'logs')
    p.add_argument('--model-dir', type=Path, default=ROOT / 'models')
    p.add_argument('--resume', type=Path, help='PPO zip; total-timesteps means additional steps')
    for name in ['final_learning_rate', 'final_clip_range', 'final_ent_coef', 'clip_range_vf']:
        p.add_argument('--' + name.replace('_', '-'), type=float, default=defaults.get(name))
    p.add_argument('--separate-critic', action=argparse.BooleanOptionalAction,
                   default=defaults.get('separate_critic', False),
                   help='Independent actor/critic CNNs; migration copies weights and resets Adam')
    p.add_argument('--reset-timesteps', action=argparse.BooleanOptionalAction,
                   default=defaults.get('reset_timesteps', False))
    p.add_argument('--early-stop-patience', type=int, default=defaults.get('early_stop_patience', 0))
    p.add_argument('--paddle-hit-bonus', type=float, default=defaults.get('paddle_hit_bonus', 0.0))
    p.add_argument('--life-loss-penalty', type=float, default=defaults.get('life_loss_penalty', 0.0))
    p.add_argument('--ppo-variant', choices=['standard', 'balanced', 'dual_rate', 'adaptive_kl'], default=defaults.get('ppo_variant', 'standard'))
    p.add_argument('--critic-max-grad-norm', type=float, default=defaults.get('critic_max_grad_norm', 1.0))
    p.add_argument('--value-loss', choices=['mse', 'huber'], default=defaults.get('value_loss', 'huber'))
    p.add_argument('--huber-delta', type=float, default=defaults.get('huber_delta', 10.0))
    p.add_argument('--critic-lr-multiplier', type=float, default=defaults.get('critic_lr_multiplier', 3.0))
    p.add_argument('--extra-critic-epochs', type=int, default=defaults.get('extra_critic_epochs', 2))
    p.add_argument('--adaptive-kl-enabled', action=argparse.BooleanOptionalAction,
                   default=defaults.get('adaptive_kl_enabled', True))
    for name, value in [('adaptive_kl_target', .0005), ('adaptive_kl_band', 1.5),
                        ('adaptive_kl_factor', 1.1), ('adaptive_kl_alpha', .2),
                        ('actor_lr_min_multiplier', .5), ('actor_lr_max_multiplier', 2.)]:
        p.add_argument('--' + name.replace('_', '-'), type=float, default=defaults.get(name, value))
    p.add_argument('--reset-optimizer', action=argparse.BooleanOptionalAction,
                   default=defaults.get('reset_optimizer', False))
    p.add_argument('--eval-envs', type=int, default=defaults.get('eval_envs', 1),
                   help='Batch development evaluation over independent seeded environments')
    p.add_argument('--fixed-eval-seeds', action=argparse.BooleanOptionalAction,
                   default=defaults.get('fixed_eval_seeds', False))
    p.add_argument('--checkpoint-selection', choices=['mean', 'stable_near_mean'],
                   default=defaults.get('checkpoint_selection', 'mean'))
    p.add_argument('--selection-mean-tolerance', type=float, default=defaults.get('selection_mean_tolerance', .05))
    p.add_argument('--post-eval-episodes', type=int, default=defaults.get('post_eval_episodes', 0),
                   help='Automatically evaluate the selected checkpoint on CPU after training; 0 disables')
    p.add_argument('--post-eval-seed', type=int, default=defaults.get('post_eval_seed', 81000))
    p.add_argument('--evaluation-dir', type=Path, default=ROOT / 'results/evaluation')
    p.add_argument('--evaluate-at-start', action=argparse.BooleanOptionalAction,
                   default=defaults.get('evaluate_at_start', True))
    args = p.parse_args(argv)
    if min(args.total_timesteps, args.n_envs, args.n_steps, args.n_epochs, args.threads,
           args.checkpoint_freq, args.eval_freq, args.n_eval_episodes) <= 0:
        p.error('Step counts, environment count, frequencies and threads must be positive')
    rollout_size = args.n_envs * args.n_steps
    if args.batch_size < 2 or rollout_size % args.batch_size:
        p.error('batch-size must be >= 2 and divide n-envs * n-steps')
    if not (0 < args.gamma <= 1 and 0 <= args.gae_lambda <= 1 and 0 < args.clip_range < 1):
        p.error('Require gamma in (0, 1], gae-lambda in [0, 1], clip-range in (0, 1)')
    if not (args.learning_rate > 0 and args.max_grad_norm > 0 and
            args.ent_coef >= 0 and args.vf_coef >= 0 and
            (args.target_kl is None or args.target_kl > 0)):
        p.error('Invalid learning rate, loss coefficient, gradient norm or KL target')
    for initial, final, label in [(args.learning_rate, args.final_learning_rate, 'learning-rate'),
                                  (args.clip_range, args.final_clip_range, 'clip-range')]:
        if final is not None and not 0 < final <= initial:
            p.error(f'final-{label} must be positive and no greater than {label}')
    if args.final_ent_coef is not None and not 0 <= args.final_ent_coef <= args.ent_coef:
        p.error('final-ent-coef must be between 0 and ent-coef')
    if args.clip_range_vf is not None and not args.clip_range_vf > 0:
        p.error('clip-range-vf must be positive')
    if args.resume and not args.reset_timesteps and (args.final_learning_rate is not None or
                                                    args.final_clip_range is not None):
        p.error('Stage learning-rate/clip schedules on resume require --reset-timesteps')
    if args.early_stop_patience < 0 or (args.early_stop_patience and not args.evaluate_at_start):
        p.error('Early stopping requires nonnegative patience and --evaluate-at-start')
    if any(not math.isfinite(value) or value < 0 for value in
           (args.paddle_hit_bonus, args.life_loss_penalty)):
        p.error('Training reward coefficients must be finite and nonnegative')
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.critic_max_grad_norm, args.huber_delta)):
        p.error('Gradient norm and Huber delta must be finite and positive')
    if not 1 <= args.eval_envs <= args.n_eval_episodes:
        p.error('eval-envs must be between 1 and n-eval-episodes')
    if args.fixed_eval_seeds and args.n_eval_episodes % args.eval_envs:
        p.error('Fixed-seed evaluation requires n-eval-episodes divisible by eval-envs')
    if not math.isfinite(args.selection_mean_tolerance) or not 0 <= args.selection_mean_tolerance < 1:
        p.error('selection-mean-tolerance must be finite and in [0, 1)')
    if args.post_eval_episodes < 0 or args.post_eval_seed < 0:
        p.error('Post-evaluation episode count and seed must be nonnegative')
    if not math.isfinite(args.critic_lr_multiplier) or args.critic_lr_multiplier <= 0 or args.extra_critic_epochs < 0:
        p.error('Critic LR multiplier must be finite and positive; extra epochs must be nonnegative')
    try:
        validate_controller(args.adaptive_kl_target, args.adaptive_kl_band, args.adaptive_kl_factor,
                            args.adaptive_kl_alpha, args.actor_lr_min_multiplier, args.actor_lr_max_multiplier)
    except ValueError as error:
        p.error(str(error))
    if args.ppo_variant == 'adaptive_kl' and (args.critic_lr_multiplier != 1. or args.extra_critic_epochs != 0):
        p.error('Adaptive KL requires --critic-lr-multiplier 1 --extra-critic-epochs 0')
    return args


def main():
    args = parse_args()
    device = setup(args.seed, args.device, args.threads)
    name = args.run_name or datetime.now().strftime('ppo_custom_fixed_angle_%Y%m%d_%H%M%S')
    log_dir, model_dir = args.log_dir / name, args.model_dir / name
    log_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    spec = environment_spec()
    metadata = dict(algorithm='ppo', arguments=vars(args), device=device,
                    versions=versions(), environment=spec)
    write_json(log_dir / 'config.json', metadata)
    state = RunState(log_dir, args.total_timesteps)
    state.update(None, 'initializing')
    env = eval_env = model = None
    started = time.monotonic()

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous_term = signal.signal(signal.SIGTERM, interrupt)
    try:
        env = make_env(args.seed, log_dir / 'train.monitor.csv', n_envs=args.n_envs,
                       paddle_hit_bonus=args.paddle_hit_bonus, life_loss_penalty=args.life_loss_penalty)
        eval_env = make_env(args.eval_seed, log_dir / 'eval/eval.monitor.csv', n_envs=args.eval_envs,
                            fixed_episode_seeds=args.fixed_eval_seeds)
        kwargs = {key: getattr(args, key) for key in [
            'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda',
            'clip_range', 'ent_coef', 'vf_coef', 'max_grad_norm', 'target_kl']}
        kwargs['clip_range_vf'] = args.clip_range_vf
        if args.final_learning_rate is not None:
            kwargs['learning_rate'] = LinearSchedule(args.learning_rate, args.final_learning_rate, 1.0)
        if args.final_clip_range is not None:
            kwargs['clip_range'] = LinearSchedule(args.clip_range, args.final_clip_range, 1.0)
        reset_optimizer = False
        balanced_kwargs = dict(critic_max_grad_norm=args.critic_max_grad_norm,
                               value_loss_type=args.value_loss, huber_delta=args.huber_delta)
        dual_kwargs = dict(critic_lr_multiplier=args.critic_lr_multiplier,
                           extra_critic_epochs=args.extra_critic_epochs)
        adaptive_kwargs = {key: getattr(args, key) for key in [
            'adaptive_kl_enabled', 'adaptive_kl_target', 'adaptive_kl_band', 'adaptive_kl_factor',
            'adaptive_kl_alpha', 'actor_lr_min_multiplier', 'actor_lr_max_multiplier']}
        if args.resume:
            model = load_model(args.resume, env=env, device=device, seed=args.seed, **kwargs)
            if not isinstance(model, PPO):
                raise ValueError('--resume requires a PPO checkpoint')
            if args.ppo_variant == 'adaptive_kl':
                if not isinstance(model, AdaptiveKLPPO):
                    model = migrate_to_adaptive_kl(model, env, device=device, seed=args.seed,
                        verbose=1, **adaptive_kwargs, **balanced_kwargs, **kwargs)
                    reset_optimizer = True
                else:
                    for key, value in dict(balanced_kwargs, **adaptive_kwargs).items():
                        setattr(model, key, value)
            elif args.ppo_variant == 'dual_rate':
                if not isinstance(model, DualRatePPO) or isinstance(model, AdaptiveKLPPO):
                    model = migrate_to_dual_rate(model, env, device=device, seed=args.seed,
                        verbose=1, **dual_kwargs, **balanced_kwargs, **kwargs)
                    reset_optimizer = True
                else:
                    for key, value in dict(balanced_kwargs, **dual_kwargs).items():
                        setattr(model, key, value)
            elif args.ppo_variant == 'balanced':
                if isinstance(model, DualRatePPO):
                    model = migrate_to_balanced(model, env, device=device, seed=args.seed,
                        verbose=1, **balanced_kwargs, **kwargs)
                    reset_optimizer = True
                elif not isinstance(model, BalancedPPO):
                    model = migrate_to_balanced(model, env, device=device, seed=args.seed,
                                                verbose=1, **balanced_kwargs, **kwargs)
                    reset_optimizer = True
                else:
                    for key, value in balanced_kwargs.items():
                        setattr(model, key, value)
            elif isinstance(model, BalancedPPO):
                raise ValueError('A custom PPO resume requires balanced, dual_rate or adaptive_kl')
            elif args.separate_critic and model.policy.share_features_extractor:
                model = split_actor_critic(model, env, device=device, seed=args.seed, verbose=1, **kwargs)
                reset_optimizer = True
        else:
            model_class = {'standard': PPO, 'balanced': BalancedPPO, 'dual_rate': DualRatePPO,
                           'adaptive_kl': AdaptiveKLPPO}[args.ppo_variant]
            extras = dict(balanced_kwargs, **adaptive_kwargs) if args.ppo_variant == 'adaptive_kl' else (
                dict(balanced_kwargs, **dual_kwargs) if args.ppo_variant == 'dual_rate' else (
                balanced_kwargs if args.ppo_variant == 'balanced' else {})
            )
            model = model_class('CnnPolicy', env, device=device, seed=args.seed, verbose=1,
                policy_kwargs=dict(share_features_extractor=not (args.separate_critic or args.ppo_variant != 'standard')),
                **extras, **kwargs)
        if args.reset_optimizer and not reset_optimizer:
            if isinstance(model, DualRatePPO):
                model.reset_optimizer()
            else:
                model.policy.optimizer = model.policy.optimizer_class(model.policy.parameters(),
                    lr=model.lr_schedule(1.0), **model.policy.optimizer_kwargs)
            reset_optimizer = True
        if isinstance(model, AdaptiveKLPPO) and reset_optimizer:
            model.reset_controller()
        model.breakout_algorithm = 'ppo'
        model.breakout_ppo_variant = args.ppo_variant
        model.breakout_env_spec = spec
        model.breakout_training_reward = dict(paddle_hit_bonus=args.paddle_hit_bonus,
                                             life_loss_penalty=args.life_loss_penalty,
                                             monitor_reward='native', evaluation_reward='native')
        # Constructors used by migrations consume RNG while allocating weights.
        # Restart both experiment arms from the requested sampling seed.
        model.set_random_seed(args.seed)
        source_steps, start_updates = model.num_timesteps, model._n_updates
        start_steps = 0 if args.reset_timesteps else source_steps
        state.target = start_steps + args.total_timesteps
        metadata.update(start_timesteps=start_steps, source_timesteps=source_steps,
                        ppo_variant=args.ppo_variant,
                        training_reward=model.breakout_training_reward,
                        optimizer_reset=reset_optimizer,
                        separate_critic=not model.policy.share_features_extractor,
                        rollout_size=args.n_envs * args.n_steps)
        write_json(log_dir / 'config.json', metadata)
        model.set_logger(configure(str(log_dir), ['stdout', 'csv', 'tensorboard']))
        stopper = (StopTrainingOnNoModelImprovement(args.early_stop_patience - 1,
                   min_evals=1, verbose=1) if args.early_stop_patience else None)
        evaluation = TrackedEvalCallback(eval_env, state=state, eval_seed=args.eval_seed,
            fixed_episode_seeds=args.fixed_eval_seeds,
            callback_after_eval=stopper,
            evaluate_at_start=args.evaluate_at_start,
            best_model_save_path=str(model_dir / 'best_model'),
            stable_model_save_path=model_dir / 'stable_model', log_path=str(log_dir / 'eval'),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=args.n_eval_episodes, deterministic=True)
        callbacks = [CheckpointCallback(save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                        save_path=str(model_dir / 'checkpoints'), name_prefix='ppo_breakout'),
                     evaluation, HealthCallback(state)]
        if args.final_ent_coef is not None:
            callbacks.append(EntropySchedule(args.ent_coef, args.final_ent_coef, args.total_timesteps))
        print(f'Device: {device}; rollout: {args.n_envs} x {args.n_steps}; target: {state.target}', flush=True)
        model.learn(args.total_timesteps, callback=callbacks,
                    reset_num_timesteps=not bool(args.resume) or args.reset_timesteps, log_interval=1)
        stopped_early = bool(stopper and stopper.no_improvement_evals >= args.early_stop_patience)
        status = 'early_stopped' if stopped_early else 'completed'
        model.save(model_dir / 'final_model')
        # Periodic on-step evaluation precedes PPO optimization. Explicitly
        # evaluate the final updated policy, including budgets shorter than eval_freq.
        if not stopped_early:
            # Learning is already finished; this measurement must not consume
            # early-stopping patience or print a spurious stop notification.
            evaluation.callback = None
            evaluation.n_calls = ((evaluation.n_calls // evaluation.eval_freq) + 1) * evaluation.eval_freq
            evaluation.num_timesteps = model.num_timesteps
            evaluation._on_step()
        selected, selection = choose_checkpoint(model_dir, args.checkpoint_selection, args.selection_mean_tolerance)
        write_json(log_dir / 'selected_model.json', dict(model=str(selected), **selection))
        write_json(log_dir / 'training_summary.json', dict(
            algorithm='ppo', status=status, target_timesteps=state.target,
            timesteps=model.num_timesteps, stage_timesteps=model.num_timesteps-start_steps,
            source_timesteps=source_steps, optimizer_reset=reset_optimizer,
            ppo_variant=args.ppo_variant,
            training_reward=model.breakout_training_reward,
            separate_critic=not model.policy.share_features_extractor,
            optimization_epochs=model._n_updates, stage_optimization_epochs=model._n_updates-start_updates,
            elapsed_seconds=time.monotonic()-started, evaluation_seconds=state.eval_seconds,
            device=str(model.device), environment=spec,
            final_model=str(model_dir / 'final_model.zip'), best_eval_reward=evaluation.best_mean_reward,
            selected_model=str(selected), checkpoint_selection=selection))
        state.update(model, status)
    except KeyboardInterrupt:
        if model is not None:
            model.save(model_dir / 'interrupted_model')
        write_json(log_dir / 'training_summary.json', dict(algorithm='ppo', status='interrupted',
            timesteps=model.num_timesteps if model else 0))
        state.update(model, 'interrupted')
        raise
    except Exception as error:
        state.update(model, 'failed', error=str(error))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()
    # Keep training completion intact if a separate held-out evaluation fails.
    if args.post_eval_episodes:
        from evaluate import run_episodes
        output = args.evaluation_dir / name / 'ppo_results.csv'
        state.update(model, 'testing', selected_model=str(selected),
                     test_episodes_completed=0, test_episodes_target=args.post_eval_episodes)
        try:
            test_model = load_model(selected, device='cpu')
            def test_progress(row, completed):
                state.update(model, 'testing', selected_model=str(selected),
                    test_episodes_completed=completed, test_episodes_target=args.post_eval_episodes)
            rows = run_episodes(test_model, args.post_eval_episodes, args.post_eval_seed,
                                output=output, on_episode=test_progress)
            meta = json.loads(output.with_suffix('.json').read_text())
            meta.update(model=str(selected.resolve()), training_timesteps=test_model.num_timesteps,
                        ppo_variant=args.ppo_variant, device='cpu',
                        checkpoint_selection=selection,
                        training_reward=getattr(test_model, 'breakout_training_reward', None))
            write_json(output.with_suffix('.json'), meta)
            write_json(log_dir / 'post_evaluation.json', dict(status='completed',
                seed=args.post_eval_seed, summary=summary(rows), selected_model=str(selected),
                output=str(output)))
            state.update(model, status, heldout_mean_reward=summary(rows)['mean_reward'],
                         heldout_results=str(output))
        except KeyboardInterrupt:
            state.update(model, 'testing_interrupted', training_status=status)
            raise
        except Exception as error:
            state.update(model, 'testing_failed', training_status=status, error=str(error))
            raise


if __name__ == '__main__':
    main()
