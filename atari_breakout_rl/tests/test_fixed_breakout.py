"""Constant-speed physics, real episode boundaries and SB3 integration."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import DQN
from env import CUSTOM_SPEC, environment_spec, make_env
from fixed_breakout import AutoFire, FixedSpeedBreakout, RepeatAndPool
from utils import setup


class FixedBreakoutTest(unittest.TestCase):
    def setUp(self):
        self.env = FixedSpeedBreakout(render_mode='rgb_array')
        self.env.reset(seed=123)

    def aim(self, position, direction):
        self.env.ball[:] = position
        self.env._direction(*direction)
        self.env.waiting_for_fire = False

    def test_gym_api_and_seed(self):
        check_env(self.env, skip_render_check=True)
        other = FixedSpeedBreakout()
        a, b = AutoFire(self.env), AutoFire(other)
        np.testing.assert_array_equal(a.reset(seed=7)[0], b.reset(seed=7)[0])
        for action in [0, 1, 2, 3] * 200:
            left, right = a.step(action), b.step(action)
            np.testing.assert_array_equal(left[0], right[0])
            self.assertEqual(left[1:], right[1:])
            if left[2] or left[3]:
                break

    def test_collisions_and_every_brick_reward(self):
        self.aim([9.5, 110], [-1, 0])
        self.env.step(0)
        self.assertGreater(self.env.velocity[0], 0)
        self.aim([80, 33.5], [0, -1])
        self.env.step(0)
        self.assertGreater(self.env.velocity[1], 0)
        self.assertEqual(self.env.paddle_width, 16)
        for row, points in enumerate([7, 7, 4, 4, 1, 1]):
            self.env.reset(seed=123)
            # Clear other bricks, retaining a sentinel so this is not wall clear.
            self.env.bricks[:] = False
            self.env.bricks[row*18+8] = self.env.bricks[17] = True
            self.aim([75, 63+row*5], [0, -1])
            _, reward, done, _, _ = self.env.step(0)
            self.assertEqual(reward, points)
            self.assertFalse(self.env.bricks[row*18+8])
            self.assertGreater(self.env.velocity[1], 0)
            self.assertFalse(done)
            self.assertAlmostEqual(np.linalg.norm(self.env.velocity), 2)

    def test_repeated_paddle_hits_never_accelerate(self):
        self.env.bricks[:] = False
        self.env.bricks[0] = True
        for hit in range(1, 101):
            self.aim([self.env.paddle_x + (hit % 5 - 2)*3, 187], [0.2, 1])
            self.env.step(0)
            self.assertEqual(self.env.paddle_hits, hit)
            self.assertLess(self.env.velocity[1], 0)
            self.assertAlmostEqual(np.linalg.norm(self.env.velocity), 2, places=12)

    def test_fixed_width_after_repeated_ceiling_contacts(self):
        for _ in range(100):
            self.aim([80, 33.5], [0, -1])
            obs, _, _, _, info = self.env.step(0)
            self.assertEqual(info['paddle_width'], 16)
            self.assertEqual(np.count_nonzero(obs[190, :, 0] == 210), 16)
            self.assertAlmostEqual(info['ball_speed'], 2)
        # Old rules must be rejected instead of being replayed as current rules.
        for version in (1, 2):
            with self.assertRaises(ValueError):
                FixedSpeedBreakout(physics_version=version)
            old_spec = dict(CUSTOM_SPEC, physics_version=version)
            with self.assertRaises(ValueError):
                environment_spec(SimpleNamespace(breakout_env_spec=old_spec))
        with self.assertRaises(ValueError):
            environment_spec(SimpleNamespace())

    def test_rebound_is_independent_of_hit_count(self):
        # Real paddle contacts at all five regions, both incoming directions,
        # including all former angle-change thresholds and a long rally.
        for dx in (-0.2, 0.2):
            for offset in (-6, -3, 0, 3, 6):
                reference = None
                for previous_hits in (0, 1, 2, 3, 6, 7, 10, 11, 12, 99):
                    self.env.paddle_hits = previous_hits
                    self.aim([self.env.paddle_x + offset, 187], [dx, 1])
                    self.env.step(0)
                    self.assertEqual(self.env.paddle_hits, previous_hits + 1)
                    self.assertAlmostEqual(np.linalg.norm(self.env.velocity), 2)
                    self.assertEqual(self.env.paddle_width, 16)
                    if reference is None:
                        reference = self.env.velocity.copy()
                    np.testing.assert_allclose(self.env.velocity, reference, atol=1e-12, rtol=0)
    def test_life_loss_and_terminal(self):
        wrapped = AutoFire(self.env)
        for lives in range(4, -1, -1):
            self.aim([20, 210], [0, 1])
            _, reward, done, truncated, info = wrapped.step(0)
            self.assertEqual(info['paddle_width'], 16)
            self.assertEqual(info['lives'], lives)
            self.assertEqual(reward, 0)
            self.assertEqual(done, lives == 0)
            self.assertFalse(truncated)
            if lives:
                _, _, done, _, info = wrapped.step(0)
                self.assertTrue(info['auto_fire'])
                self.assertFalse(self.env.waiting_for_fire)
                self.assertFalse(done)
        with self.assertRaises(RuntimeError):
            wrapped.step(0)

    def test_two_walls_and_timeout(self):
        self.env.score = 431
        for wall in (1, 2):
            self.env.bricks[:] = False
            self.env.bricks[98] = True
            self.aim([75, 88], [0, -1])
            _, reward, done, _, _ = self.env.step(0)
            self.assertEqual(self.env.paddle_width, 16)
            self.assertEqual(reward, 1)
            self.assertEqual(done, wall == 2)
            if wall == 1:
                self.assertEqual(self.env.score, 432)
                self.assertEqual(self.env.wall, 2)
                self.assertTrue(self.env.bricks.all())
                self.env.score = 863
        self.assertEqual(self.env.score, 864)
        timeout = RepeatAndPool(AutoFire(FixedSpeedBreakout(max_frames=1)))
        timeout.reset(seed=3)
        obs, _, done, truncated, _ = timeout.step(0)
        self.assertFalse(done)
        self.assertTrue(truncated)
        np.testing.assert_array_equal(obs, timeout.unwrapped.render())

    def test_long_trajectory_speed_and_frame_capture(self):
        game = AutoFire(self.env)
        # Follow the ball only as a physics test, never as a learned policy.
        rewards, hits, measured = 0, 0, []
        for step in range(20000):
            dx = self.env.ball[0] - self.env.paddle_x
            action = 2 if dx > 1.5 else 3 if dx < -1.5 else 0
            _, reward, done, truncated, info = game.step(action)
            self.assertEqual(info['paddle_width'], 16)
            rewards += reward
            hits = max(hits, info['paddle_hits'])
            if not info['waiting_for_fire'] and not done:
                measured.append(info['ball_speed'])
            if done or truncated:
                break
        self.assertGreater(hits, 12)
        self.assertGreater(rewards, 20)
        np.testing.assert_allclose(measured, 2, atol=1e-12, rtol=0)

    def test_sb3_contract_roundtrip_and_display(self):
        setup(123, 'cpu', 1)
        captured = []
        env = make_env(123)
        shown = make_env(123, frame_callback=lambda *x: captured.append(x))
        try:
            np.testing.assert_array_equal(env.reset(), shown.reset())
            captured.clear()
            for action in [0, 1, 2, 3]*10:
                a, b = env.step([action]), shown.step([action])
                for i in range(3):
                    np.testing.assert_array_equal(a[i], b[i])
            self.assertEqual(len(captured), 160)
            self.assertEqual(a[0].shape, (1, 4, 84, 84))
            model = DQN('CnnPolicy', env, buffer_size=64, learning_starts=8,
                        batch_size=8, seed=123, device='cpu')
            model.breakout_env_spec = environment_spec()
            model.learn(32)
            self.assertGreater(model._n_updates, 0)
            with tempfile.TemporaryDirectory() as directory:
                model.save(Path(directory)/'custom')
                loaded = DQN.load(Path(directory)/'custom', device='cpu')
                self.assertEqual(environment_spec(loaded), CUSTOM_SPEC)
                np.testing.assert_array_equal(model.predict(a[0])[0], loaded.predict(a[0])[0])
        finally:
            env.close()
            shown.close()


if __name__ == '__main__':
    unittest.main()
