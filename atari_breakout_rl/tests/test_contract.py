"""Integration checks for the highest-risk Atari/SB3 boundaries."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from stable_baselines3 import DQN
from env import make_env
from utils import setup


class EnvironmentContract(unittest.TestCase):
    def setUp(self):
        setup(123, 'cpu', 1)
        self.env = make_env(123, render_mode='rgb_array')

    def tearDown(self):
        self.env.close()

    def test_preprocessing_and_seed(self):
        self.env.seed(123)
        first = self.env.reset().copy()
        self.env.seed(123)
        second = self.env.reset()
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (1, 4, 84, 84))
        self.assertEqual(first.dtype, np.uint8)
        self.assertEqual(self.env.get_images()[0].shape, (210, 160, 3))
        meanings = self.env.env_method('get_action_meanings')[0]
        self.assertEqual(len(meanings), self.env.action_space.n)
        self.assertIn('FIRE', meanings)
        # Frame stack must move the last three channels into the first three slots.
        before = self.env.venv.venv.envs[0].unwrapped.ale.getFrameNumber()
        following, _, _, _ = self.env.step([meanings.index('RIGHT')])
        after = self.env.venv.venv.envs[0].unwrapped.ale.getFrameNumber()
        self.assertEqual(after - before, 4)
        np.testing.assert_array_equal(following[:, :3], second[:, 1:])

    def test_model_round_trip(self):
        model = DQN('CnnPolicy', self.env, buffer_size=64, learning_starts=8,
                    batch_size=8, seed=123, device='cpu')
        model.learn(32)
        self.assertGreater(model._n_updates, 0)
        obs = self.env.reset()
        expected = model.predict(obs, deterministic=True)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test_model'
            model.save(path)
            loaded = DQN.load(path, env=self.env, device='cpu')
            np.testing.assert_array_equal(expected, loaded.predict(obs, deterministic=True)[0])
            self.assertEqual(model.num_timesteps, loaded.num_timesteps)


if __name__ == '__main__':
    unittest.main()
