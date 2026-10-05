"""Verify SB3 n-step activation and safe compressed replay continuation."""
from pathlib import Path
import sys
import tempfile
import unittest

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import NStepReplayBuffer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from replay_io import load_replay, save_replay
from env import environment_spec


class ReplayPersistence(unittest.TestCase):
    def test_nstep_migration_compression_and_episode_boundary(self):
        env = gym.make('CartPole-v1')
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = DQN('MlpPolicy', env, buffer_size=32, device='cpu')
                source.breakout_env_spec = environment_spec()
                source.save(root / 'source')
                model = DQN.load(root / 'source', env=env, device='cpu', n_steps=3,
                                 replay_buffer_class=None, replay_buffer_kwargs={})
                self.assertIsInstance(model.replay_buffer, NStepReplayBuffer)
                self.assertEqual(model.replay_buffer.n_steps, 3)
                replay = model.replay_buffer
                for i in range(6):
                    replay.add(np.full((1, 4), i, dtype=np.float32),
                               np.full((1, 4), i + 1, dtype=np.float32),
                               np.array([[0]]), np.array([1.0]), np.array([False]), [{}])
                sample = replay._get_samples(np.array([0]))
                self.assertAlmostEqual(sample.rewards.item(), 1 + model.gamma + model.gamma ** 2, places=5)
                self.assertAlmostEqual(sample.discounts.item(), model.gamma ** 3, places=5)
                path = save_replay(model, root / 'replay.pkl', compress=True)
                self.assertEqual(path.suffix, '.gz')
                self.assertFalse(path.with_name(path.name + '.tmp').exists())
                restored = DQN('MlpPolicy', env, buffer_size=32, n_steps=3, device='cpu')
                restored.breakout_env_spec = environment_spec()
                load_replay(restored, path)
                self.assertEqual(restored.replay_buffer.size(), 6)
                np.testing.assert_array_equal(restored.replay_buffer.observations, replay.observations)
                self.assertEqual(restored.replay_buffer.dones[5, 0], 1)
                self.assertEqual(restored.replay_buffer.timeouts[5, 0], 1)
                # New environment transitions must not enter the old episode's return.
                restored.replay_buffer.add(np.full((1, 4), 100, dtype=np.float32),
                    np.full((1, 4), 101, dtype=np.float32), np.array([[0]]),
                    np.array([100.0]), np.array([False]), [{}])
                boundary = restored.replay_buffer._get_samples(np.array([5]))
                self.assertAlmostEqual(boundary.rewards.item(), 1)
                self.assertAlmostEqual(boundary.discounts.item(), model.gamma, places=5)
                self.assertEqual(boundary.dones.item(), 0)  # timeout still bootstraps
                self.assertEqual(boundary.next_observations[0, 0].item(), 6)
                with self.assertRaisesRegex(ValueError, 'requested 1'):
                    load_replay(source, path)
                plain = save_replay(source, root / 'vanilla.pkl')
                legacy = DQN('MlpPolicy', env, buffer_size=32, device='cpu')
                legacy.breakout_env_spec = environment_spec()
                load_replay(legacy, plain)
                self.assertNotIsInstance(legacy.replay_buffer, NStepReplayBuffer)
                self.assertEqual(legacy.replay_buffer.size(), source.replay_buffer.size())
                with self.assertRaisesRegex(ValueError, 'requested 3'):
                    load_replay(restored, plain)
                # Matching tensor shapes cannot make an old physics version safe.
                restored.breakout_env_spec = dict(environment_spec(), physics_version=2)
                original_replay = restored.replay_buffer
                with self.assertRaisesRegex(ValueError, 'physics_version'):
                    load_replay(restored, path)
                self.assertIs(restored.replay_buffer, original_replay)
                restored.breakout_env_spec = environment_spec()
                load_replay(restored, path)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
