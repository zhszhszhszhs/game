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

    def test_lost_life_restarts_game_without_ending_episode(self):
        observation = self.env.reset()
        initial_lives = self.env.venv.venv.envs[0].unwrapped.ale.lives()
        recovered = False
        noop = self.env.env_method('get_action_meanings')[0].index('NOOP')
        for _ in range(2000):
            observation, rewards, dones, infos = self.env.step([noop])  # NOOP after reset FIRE
            self.assertFalse(dones[0], 'One lost life must not end a complete game')
            if infos[0].get('ale_life_lost'):
                self.assertTrue(infos[0]['ale_auto_fire'])
                self.assertEqual(infos[0]['lives'], initial_lives - 1)
                recovered = True
                break
        self.assertTrue(recovered, 'ALE Breakout did not lose a life in 2000 NOOP steps')
        self.assertEqual(observation.shape, (1, 4, 84, 84))
        # A returned frame after FIRE means the environment can continue serving.
        self.assertEqual(self.env.venv.venv.envs[0].unwrapped.ale.lives(), initial_lives - 1)
        for _ in range(2000):
            _, _, dones, infos = self.env.step([noop])
            if dones[0]:
                self.assertFalse(infos[0]['TimeLimit.truncated'])
                self.assertEqual(infos[0]['lives'], 0)
                break
        else:
            self.fail('Repeated NOOP must finish a real game, not stay waiting for FIRE')

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

    def test_display_frames_preserve_game_transitions(self):
        frames = []
        collecting = False

        def capture(frame, reward, lives):
            nonlocal collecting
            if frame is None:
                collecting = False
            elif collecting:
                frames.append((frame.copy(), reward, lives))

        observed = make_env(123, render_mode='rgb_array', frame_callback=capture)
        try:
            np.testing.assert_array_equal(self.env.reset(), observed.reset())
            for _ in range(2000):
                frames.clear()
                collecting = True
                expected = self.env.step([0])
                actual = observed.step([0])
                np.testing.assert_array_equal(expected[0], actual[0])
                np.testing.assert_array_equal(expected[1], actual[1])
                np.testing.assert_array_equal(expected[2], actual[2])
                self.assertGreater(len(frames), 0)
                self.assertLessEqual(len(frames), 8)
                self.assertEqual(frames[-1][0].shape, (210, 160, 3))
                self.assertAlmostEqual(sum(f[1] for f in frames), float(actual[1][0]))
                if actual[2][0]:
                    # Last displayed frame must belong to the finished game,
                    # not to the VecEnv's automatic reset or its FIRE actions.
                    self.assertEqual(frames[-1][2], 0)
                    self.assertFalse(collecting)
                    break
            else:
                self.fail('Expected a complete game')
        finally:
            observed.close()


if __name__ == '__main__':
    unittest.main()
