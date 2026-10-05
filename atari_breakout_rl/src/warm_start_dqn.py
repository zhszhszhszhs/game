"""Keep a trained epsilon-greedy policy while filling fresh replay.

Only action collection during warm-up changes. SB3 still owns the CNN,
Bellman targets, replay sampling, optimizer, and learning-start threshold.
"""
from stable_baselines3 import DQN


class WarmStartDQN(DQN):
    policy_warmup = False

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        # Do not modify self.learning_starts: gradient updates must still wait
        # for the configured amount of fresh experience. predict(False) in SB3
        # retains its normal epsilon-greedy exploration.
        threshold = 0 if self.policy_warmup else learning_starts
        return super()._sample_action(threshold, action_noise, n_envs)
