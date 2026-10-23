"""Training health and evaluation accounting for SB3 DQN and PPO."""
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import os
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from utils import write_csv, write_json
from replay_io import save_replay


class RunState:
    def __init__(self, directory, target):
        self.path = directory / 'runtime.json'
        self.started = time.monotonic()
        self.target = target
        self.eval_seconds = 0.0

    def update(self, model, status='training', **extra):
        elapsed = time.monotonic() - self.started
        write_json(self.path, dict(status=status, pid=os.getpid(),
            timesteps=model.num_timesteps if model else 0, target_timesteps=self.target,
            elapsed_seconds=elapsed, evaluation_seconds=self.eval_seconds,
            sampling_and_training_seconds=max(0, elapsed - self.eval_seconds),
            updated_at=datetime.now(timezone.utc).isoformat(), **extra))


class HealthCallback(BaseCallback):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.timeouts = deque(maxlen=100)
        self.auto_fires = 0

    def _on_step(self):
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            self.auto_fires += int(info.get('auto_fire', False))
            if done and 'episode' in info:
                self.timeouts.append(bool(info.get('TimeLimit.truncated', False)))
        if self.n_calls % 1000 == 0:
            loss = self.logger.name_to_value.get('train/loss')
            if loss is not None and not np.isfinite(loss):
                raise FloatingPointError(f'Non-finite training loss at step {self.num_timesteps}')
            self.logger.record('time/total_timesteps', self.num_timesteps)
            update_key = 'optimization_epochs' if hasattr(self.model, 'rollout_buffer') else 'gradient_updates'
            details = {update_key: self.model._n_updates}
            if hasattr(self.model, 'exploration_rate'):
                self.logger.record('rollout/exploration_rate', self.model.exploration_rate)
                details['exploration_rate'] = self.model.exploration_rate
            replay = getattr(self.model, 'replay_buffer', None)
            if replay is not None:
                self.logger.record('replay/size', replay.size())
                details['replay_size'] = replay.size()
            self.logger.record('rollout/auto_fire_count', self.auto_fires)
            self.logger.record('time/evaluation_seconds', self.state.eval_seconds)
            if self.timeouts:
                self.logger.record('rollout/timeout_rate', np.mean(self.timeouts))
            if self.model.ep_info_buffer:
                items = self.model.ep_info_buffer
                self.logger.record('rollout/ep_rew_mean', np.mean([x['r'] for x in items]))
                self.logger.record('rollout/ep_len_mean', np.mean([x['l'] for x in items]))
                shaped = [x['training_r'] for x in items if 'training_r' in x]
                if shaped:
                    self.logger.record('rollout/training_reward_mean', np.mean(shaped))
            self.logger.dump(self.num_timesteps)
            self.state.update(self.model, **details)
        return True

    def _on_training_end(self):
        # PPO performs its last optimization after the last rollout callback.
        loss = self.logger.name_to_value.get('train/loss')
        if loss is not None and not np.isfinite(loss):
            raise FloatingPointError('Non-finite final training loss')
        self.logger.dump(self.num_timesteps)


class TrackedEvalCallback(EvalCallback):
    """Repeat the same development seeds, and expose truncation and elapsed time."""
    def __init__(self, *args, state, eval_seed, stable_model_save_path=None,
                 evaluate_at_start=False, eval_start_step=0,
                 save_best_replay=False, compress_replay=False, fixed_episode_seeds=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state
        self.eval_seed = eval_seed
        self.timeouts = []
        self.stable_model_save_path = Path(stable_model_save_path) if stable_model_save_path else None
        self.evaluate_at_start = evaluate_at_start
        self.eval_start_step = eval_start_step
        self.save_best_replay = save_best_replay
        self.compress_replay = compress_replay
        self.best_stability = (float('-inf'), float('-inf'))
        self.fixed_episode_seeds = fixed_episode_seeds
        if fixed_episode_seeds and self.n_eval_episodes % self.eval_env.num_envs:
            raise ValueError('Fixed evaluation episode count must divide evenly across environments')
        self.episode_rows = []
        self.eval_transitions = 0
        self.last_eval_heartbeat = 0.0

    def _on_training_start(self):
        # Evaluate the inherited policy before updating it, so a weaker first
        # checkpoint cannot replace the starting policy as the stage's best.
        if self.evaluate_at_start:
            self._on_step()

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        self.eval_transitions += 1
        if locals_['done'] and 'episode' in locals_['info']:
            info = locals_['info']
            self.timeouts.append(bool(info.get('TimeLimit.truncated', False)))
            if self.fixed_episode_seeds:
                self.episode_rows.append(dict(timesteps=self.num_timesteps, seed=int(info['episode_seed']),
                    reward=float(info['episode']['r']), length=int(info['episode']['l']),
                    truncated=bool(info.get('TimeLimit.truncated', False))))
        now = time.monotonic()
        if locals_['done'] or now - self.last_eval_heartbeat >= 5.0:
            self.last_eval_heartbeat = now
            self.state.update(self.model, 'evaluating',
                evaluation_episodes_completed=len(self.timeouts),
                evaluation_episodes_target=self.n_eval_episodes,
                evaluation_transitions=self.eval_transitions)

    def _on_step(self):
        due = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if not due:
            return super()._on_step()
        # The initial evaluation protects the inherited policy. Repeated checks
        # during replay warm-up cannot show learning and must not consume patience.
        if self.n_calls > 0 and self.num_timesteps < self.eval_start_step:
            return True
        self.eval_env.seed(self.eval_seed)
        self.timeouts = []
        self.episode_rows = []
        self.eval_transitions = 0
        self.last_eval_heartbeat = time.monotonic()
        self.state.update(self.model, 'evaluating')
        started = time.monotonic()
        previous_best = self.best_mean_reward
        try:
            result = super()._on_step()
        finally:
            self.state.eval_seconds += time.monotonic() - started
        if self.fixed_episode_seeds:
            expected = list(range(self.eval_seed, self.eval_seed + self.n_eval_episodes))
            if sorted(row['seed'] for row in self.episode_rows) != expected:
                raise RuntimeError('Development evaluation did not use the exact episode seed list')
            if self.log_path is not None:
                directory = Path(self.log_path).parent
                # One file per evaluation also survives later interrupted evaluations.
                write_csv(directory / f'episodes_{self.num_timesteps}.csv',
                          sorted(self.episode_rows, key=lambda row: row['seed']))
        if self.timeouts:
            self.logger.record('eval/timeout_rate', np.mean(self.timeouts))
        if self.evaluations_results:
            rewards = np.asarray(self.evaluations_results[-1], dtype=float)
            median, mean = float(np.median(rewards)), float(np.mean(rewards))
            self.logger.record('eval/median_reward', median)
            self.logger.record('eval/reward_std', float(np.std(rewards)))
            self.logger.record('eval/min_reward', float(np.min(rewards)))
            self.logger.record('eval/max_reward', float(np.max(rewards)))
            self.logger.record('eval/p25_reward', float(np.percentile(rewards, 25)))
            self.logger.record('eval/fraction_below_50', float(np.mean(rewards < 50)))
            if self.best_model_save_path and self.best_mean_reward > previous_best:
                replay_path = None
                replay = getattr(self.model, 'replay_buffer', None)
                if self.save_best_replay and replay is not None and replay.size() > 0:
                    replay_path = save_replay(self.model,
                        Path(self.best_model_save_path) / 'replay_buffer.pkl',
                        self.compress_replay)
                write_json(Path(self.best_model_save_path) / 'selection.json', dict(
                    criterion='mean_reward', timesteps=self.num_timesteps,
                    mean_reward=mean, median_reward=median, std_reward=float(np.std(rewards)),
                    episodes=len(rewards), eval_seed=self.eval_seed,
                    evaluation_protocol='fixed_episode_seeds' if self.fixed_episode_seeds else 'vector_rng_stream',
                    replay_buffer_path=replay_path,
                    replay_size=replay.size() if replay is not None else None))
            if self.stable_model_save_path and (median, mean) > self.best_stability:
                self.stable_model_save_path.mkdir(parents=True, exist_ok=True)
                self.model.save(self.stable_model_save_path / 'best_model')
                self.best_stability = (median, mean)
                write_json(self.stable_model_save_path / 'selection.json', dict(
                    criterion='median_reward_then_mean_reward', timesteps=self.num_timesteps,
                    median_reward=median, mean_reward=mean, std_reward=float(np.std(rewards)),
                    episodes=len(rewards), eval_seed=self.eval_seed,
                    evaluation_protocol='fixed_episode_seeds' if self.fixed_episode_seeds else 'vector_rng_stream'))
        self.logger.record('eval/duration_seconds', time.monotonic() - started)
        self.logger.dump(self.num_timesteps)
        self.state.update(self.model, latest_eval_reward=self.last_mean_reward,
                          best_eval_reward=self.best_mean_reward,
                          best_median_reward=self.best_stability[0] if self.stable_model_save_path else None)
        return result
