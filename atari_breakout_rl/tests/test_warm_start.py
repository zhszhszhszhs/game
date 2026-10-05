"""Fresh replay should use the resumed policy without starting optimization early."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from warm_start_dqn import WarmStartDQN
from utils import setup


class WarmStartTest(unittest.TestCase):
    def test_policy_collection_delayed_updates_and_standard_loading(self):
        setup(42, 'cpu', 1)
        env = gym.make('CartPole-v1')
        try:
            model = WarmStartDQN('MlpPolicy', env, buffer_size=64, learning_starts=8,
                batch_size=4, train_freq=4, n_steps=3, device='cpu', seed=42)
            model.policy_warmup = True
            with patch.object(model, 'predict', return_value=(np.array([1]), None)) as predict:
                model.learn(8)
                self.assertEqual(predict.call_count, 8)
                self.assertTrue(all(call.kwargs['deterministic'] is False
                                    for call in predict.call_args_list))
            self.assertEqual(model.learning_starts, 8)
            self.assertEqual(model._n_updates, 0)
            self.assertEqual(model.replay_buffer.size(), 8)
            np.testing.assert_array_equal(model.replay_buffer.actions[:8], np.ones((8, 1, 1)))
            model.learn(8, reset_num_timesteps=False)
            self.assertEqual(model._n_updates, 2)
            obs = model.get_env().reset()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model'
                model.save(path)
                loaded = DQN.load(path, device='cpu')
                np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                              loaded.predict(obs, deterministic=True)[0])
        finally:
            env.close()

    def test_default_still_collects_random_warmup(self):
        setup(42, 'cpu', 1)
        env = gym.make('CartPole-v1')
        try:
            model = WarmStartDQN('MlpPolicy', env, buffer_size=64, learning_starts=8,
                                batch_size=4, device='cpu')
            with patch.object(model, 'predict', wraps=model.predict) as predict:
                model.learn(8)
                self.assertEqual(predict.call_count, 0)
            self.assertEqual(model._n_updates, 0)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
