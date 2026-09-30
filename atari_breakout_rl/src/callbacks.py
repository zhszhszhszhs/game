"""Training health and evaluation accounting; the DQN implementation stays in SB3."""
from collections import deque
from datetime import datetime, timezone
import os
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from utils import write_json


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
            self.auto_fires += int(info.get('ale_auto_fire', False))
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
    def __init__(self, *args, state, eval_seed, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state
        self.eval_seed = eval_seed
        self.timeouts = []

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        if locals_['done'] and 'episode' in locals_['info']:
            self.timeouts.append(bool(locals_['info'].get('TimeLimit.truncated', False)))

    def _on_step(self):
        due = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if not due:
            return super()._on_step()
        self.eval_env.seed(self.eval_seed)
        self.timeouts = []
        self.state.update(self.model, 'evaluating')
        started = time.monotonic()
        try:
            result = super()._on_step()
        finally:
            self.state.eval_seconds += time.monotonic() - started
        if self.timeouts:
            self.logger.record('eval/timeout_rate', np.mean(self.timeouts))
        self.logger.record('eval/duration_seconds', time.monotonic() - started)
        self.logger.dump(self.num_timesteps)
        self.state.update(self.model, latest_eval_reward=self.last_mean_reward,
                          best_eval_reward=self.best_mean_reward)
        return result
