"""Training health and evaluation accounting; the DQN implementation stays in SB3."""
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import os
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from utils import write_json
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
                raise FloatingPointError(f'Non-finite DQN loss at step {self.num_timesteps}')
            self.logger.record('time/total_timesteps', self.num_timesteps)
            self.logger.record('rollout/exploration_rate', self.model.exploration_rate)
            self.logger.record('rollout/auto_fire_count', self.auto_fires)
            self.logger.record('replay/size', self.model.replay_buffer.size())
            self.logger.record('time/evaluation_seconds', self.state.eval_seconds)
            if self.timeouts:
                self.logger.record('rollout/timeout_rate', np.mean(self.timeouts))
            if self.model.ep_info_buffer:
                items = self.model.ep_info_buffer
                self.logger.record('rollout/ep_rew_mean', np.mean([x['r'] for x in items]))
                self.logger.record('rollout/ep_len_mean', np.mean([x['l'] for x in items]))
            self.logger.dump(self.num_timesteps)
            self.state.update(self.model, exploration_rate=self.model.exploration_rate,
                gradient_updates=self.model._n_updates,
                replay_size=self.model.replay_buffer.size())
        return True


class TrackedEvalCallback(EvalCallback):
    """Repeat the same development seeds, and expose truncation and elapsed time."""
    def __init__(self, *args, state, eval_seed, stable_model_save_path=None,
                 evaluate_at_start=False, eval_start_step=0,
                 save_best_replay=False, compress_replay=False, **kwargs):
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

    def _on_training_start(self):
        # Evaluate the inherited policy before updating it, so a weaker first
        # checkpoint cannot replace the starting policy as the stage's best.
        if self.evaluate_at_start:
            self._on_step()

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        if locals_['done'] and 'episode' in locals_['info']:
            self.timeouts.append(bool(locals_['info'].get('TimeLimit.truncated', False)))

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
        self.state.update(self.model, 'evaluating')
        started = time.monotonic()
        previous_best = self.best_mean_reward
        try:
            result = super()._on_step()
        finally:
            self.state.eval_seconds += time.monotonic() - started
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
                if self.save_best_replay and self.model.replay_buffer.size() > 0:
                    replay_path = save_replay(self.model,
                        Path(self.best_model_save_path) / 'replay_buffer.pkl',
                        self.compress_replay)
                write_json(Path(self.best_model_save_path) / 'selection.json', dict(
                    criterion='mean_reward', timesteps=self.num_timesteps,
                    mean_reward=mean, median_reward=median, std_reward=float(np.std(rewards)),
                    episodes=len(rewards), eval_seed=self.eval_seed,
                    replay_buffer_path=replay_path,
                    replay_size=self.model.replay_buffer.size()))
            if self.stable_model_save_path and (median, mean) > self.best_stability:
                self.stable_model_save_path.mkdir(parents=True, exist_ok=True)
                self.model.save(self.stable_model_save_path / 'best_model')
                self.best_stability = (median, mean)
                write_json(self.stable_model_save_path / 'selection.json', dict(
                    criterion='median_reward_then_mean_reward', timesteps=self.num_timesteps,
                    median_reward=median, mean_reward=mean, std_reward=float(np.std(rewards)),
                    episodes=len(rewards), eval_seed=self.eval_seed))
        self.logger.record('eval/duration_seconds', time.monotonic() - started)
        self.logger.dump(self.num_timesteps)
        self.state.update(self.model, latest_eval_reward=self.last_mean_reward,
                          best_eval_reward=self.best_mean_reward,
                          best_median_reward=self.best_stability[0] if self.stable_model_save_path else None)
        return result
