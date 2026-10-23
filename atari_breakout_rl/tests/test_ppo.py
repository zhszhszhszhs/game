"""PPO integration: distinct vector environments, updates, checkpoint and resume."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from callbacks import HealthCallback, RunState, TrackedEvalCallback
from env import environment_spec, make_env
from evaluate import run_episodes
from game_viewer import GameViewer
from model_io import load_model
from train_ppo import parse_args
from ppo_tricks import EntropySchedule, split_actor_critic


class PPOIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_vector_seeds_and_terminal_contract(self):
        env = make_env(42, n_envs=2)
        reference = make_env(43)
        try:
            obs = env.reset()
            self.assertEqual(obs.shape, (2, 4, 84, 84))
            self.assertEqual(obs.dtype, np.uint8)
            np.testing.assert_array_equal(obs[1], reference.reset()[0])
            for wrapped in env.venv.venv.envs:
                wrapped.unwrapped.max_frames = 4
            _, _, dones, infos = env.step([0, 0])
            self.assertTrue(dones.all())
            for info in infos:
                self.assertTrue(info['TimeLimit.truncated'])
                self.assertEqual(info['terminal_observation'].shape, (4, 84, 84))
        finally:
            env.close()
            reference.close()

    def test_invalid_rollout_batch_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--n-envs', '2', '--n-steps', '8', '--batch-size', '7'])

    def test_split_critic_preserves_policy_and_isolates_value_gradients(self):
        env = make_env(42)
        try:
            source = PPO('CnnPolicy', env, n_steps=8, batch_size=8, device='cpu')
            source.num_timesteps = 123
            model = split_actor_critic(source, env, n_steps=8, batch_size=8, device='cpu')
            self.assertEqual(model.num_timesteps, 123)
            self.assertIsNot(model.policy.pi_features_extractor, model.policy.vf_features_extractor)
            obs, _ = source.policy.obs_to_tensor(env.reset())
            with torch.no_grad():
                torch.testing.assert_close(source.policy.get_distribution(obs).distribution.probs,
                                           model.policy.get_distribution(obs).distribution.probs)
                torch.testing.assert_close(source.policy.predict_values(obs), model.policy.predict_values(obs))
            actor_before = {k: v.clone() for k, v in model.policy.pi_features_extractor.state_dict().items()}
            critic_before = {k: v.clone() for k, v in model.policy.vf_features_extractor.state_dict().items()}
            model.policy.optimizer.zero_grad()
            (model.policy.predict_values(obs) - 5).square().mean().backward()
            model.policy.optimizer.step()
            for key, value in model.policy.pi_features_extractor.state_dict().items():
                torch.testing.assert_close(value, actor_before[key], rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(value, critic_before[key])
                                for key, value in model.policy.vf_features_extractor.state_dict().items()))
            with tempfile.TemporaryDirectory() as directory:
                model.breakout_algorithm = 'ppo'
                model.breakout_env_spec = environment_spec()
                path = Path(directory) / 'split.zip'
                model.save(path)
                loaded = load_model(path, device='cpu')
                self.assertIsNot(loaded.policy.pi_features_extractor, loaded.policy.vf_features_extractor)
                with torch.no_grad():
                    torch.testing.assert_close(model.policy.get_distribution(obs).distribution.probs,
                                               loaded.policy.get_distribution(obs).distribution.probs)
        finally:
            env.close()

    def test_entropy_schedule_uses_additional_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            env = make_env(42)
            try:
                model = PPO('CnnPolicy', env, n_steps=8, batch_size=8, device='cpu')
                model.set_logger(configure(directory, []))
                model.num_timesteps = 400000
                schedule = EntropySchedule(.003, .001, 100)
                schedule.init_callback(model)
                schedule.on_training_start({}, {})
                schedule.on_rollout_start()
                self.assertAlmostEqual(model.ent_coef, .003)
                model.num_timesteps += 50
                schedule.on_rollout_start()
                self.assertAlmostEqual(model.ent_coef, .002)
                model.num_timesteps += 100
                schedule.on_rollout_start()
                self.assertAlmostEqual(model.ent_coef, .001)
            finally:
                env.close()

    def test_resume_decay_requires_stage_counter(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--resume', 'model.zip', '--final-learning-rate', '0.00001'])

    def test_event_shaping_preserves_native_monitor_and_full_games(self):
        env = make_env(42, paddle_hit_bonus=.1, life_loss_penalty=.5)
        raw = env.venv.venv.envs[0].unwrapped
        try:
            env.reset()
            raw.waiting_for_fire = False
            raw.ball[:] = [80, 187]
            raw._direction(0, 1)
            _, rewards, dones, infos = env.step([0])
            self.assertFalse(dones[0])
            self.assertEqual(infos[0]['paddle_contacts'], 1)
            self.assertEqual(infos[0]['native_reward'], 0)
            self.assertAlmostEqual(float(rewards[0]), .1)
            # A lost life resets the game's per-life paddle hit counter. That
            # reset must not produce a spurious contact or terminate the game.
            raw.ball[:] = [20, 210]
            raw._direction(0, 1)
            _, rewards, dones, infos = env.step([0])
            self.assertFalse(dones[0])
            self.assertEqual(infos[0]['paddle_contacts'], 0)
            self.assertEqual(infos[0]['lives_lost'], 1)
            self.assertAlmostEqual(float(rewards[0]), -.5)
            raw.lives = 1
            raw.ball[:] = [20, 210]
            raw._direction(0, 1)
            _, rewards, dones, infos = env.step([0])
            self.assertTrue(dones[0])
            self.assertEqual(infos[0]['episode']['r'], 0)
            self.assertAlmostEqual(infos[0]['episode']['training_r'], -.9)
            self.assertEqual(infos[0]['terminal_observation'].shape, (4, 84, 84))
        finally:
            env.close()

    def test_shaping_timeout_and_native_evaluation(self):
        for bonus in [0, .1]:
            env = make_env(42, paddle_hit_bonus=bonus)
            raw = env.venv.venv.envs[0].unwrapped
            try:
                env.reset()
                raw.max_frames = 4
                raw.waiting_for_fire = False
                raw.ball[:] = [80, 187]
                raw._direction(0, 1)
                _, rewards, dones, infos = env.step([0])
                self.assertTrue(dones[0])
                self.assertTrue(infos[0]['TimeLimit.truncated'])
                self.assertAlmostEqual(float(rewards[0]), bonus)
                self.assertEqual(infos[0]['episode']['r'], 0)
                self.assertEqual('training_r' in infos[0]['episode'], bool(bonus))
            finally:
                env.close()

    def test_nonfinite_shaping_coefficients_rejected(self):
        for value in ['-1', 'nan', 'inf']:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--paddle-hit-bonus', value])

    def test_update_evaluate_save_reload_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = make_env(42, root / 'train.monitor.csv', n_envs=2)
            eval_env = make_env(31000)
            try:
                model = PPO('CnnPolicy', env, n_steps=8, batch_size=16, n_epochs=1,
                            seed=42, device='cpu')
                model.breakout_algorithm = 'ppo'
                model.breakout_env_spec = environment_spec()
                model.set_logger(configure(str(root / 'logs'), []))
                before = {key: value.clone() for key, value in model.policy.state_dict().items()}
                state = RunState(root, 32)
                health = HealthCallback(state)
                model.learn(32, callback=health)
                self.assertGreater(model._n_updates, 0)
                self.assertTrue(any(not torch.equal(before[key], value)
                                    for key, value in model.policy.state_dict().items()))
                # Exercise the shared health path without a DQN replay buffer.
                health.n_calls = 999
                self.assertTrue(health.on_step())
                runtime = json.loads(state.path.read_text())
                self.assertNotIn('replay_size', runtime)
                self.assertEqual(runtime['optimization_epochs'], model._n_updates)
                evaluation = TrackedEvalCallback(eval_env, state=state, eval_seed=31000,
                    best_model_save_path=str(root / 'best'), log_path=str(root / 'eval'),
                    n_eval_episodes=1, eval_freq=1, verbose=0)
                evaluation.init_callback(model)
                evaluation.on_step()
                selection = json.loads((root / 'best/selection.json').read_text())
                self.assertIsNone(selection['replay_size'])
                loaded = load_model(root / 'best/best_model.zip', env=env, device='cpu')
                self.assertIsInstance(loaded, PPO)
                obs = env.reset()
                np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                              loaded.predict(obs, deterministic=True)[0])
                with contextlib.redirect_stdout(io.StringIO()):
                    rows = run_episodes(loaded, episodes=1, seed=41000)
                self.assertEqual(len(rows), 1)
                self.assertEqual(loaded.n_envs, 2)
                loaded.learn(16, reset_num_timesteps=False)
                self.assertEqual(loaded.num_timesteps, 48)
                self.assertEqual(len(list(root.glob('train.*.monitor.csv'))), 2)
            finally:
                env.close()
                eval_env.close()

    def test_viewer_lists_ppo_and_excludes_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run in ['ppo_custom_fixed_angle_test', 'ppo_custom_fixed_angle_smoke_test',
                        'dqn_custom_fixed_angle_test']:
                path = root / run / 'final_model.zip'
                path.parent.mkdir()
                path.touch()
            models = GameViewer(root).models()
            self.assertIn('ppo_custom_fixed_angle_test/final_model.zip', models)
            self.assertIn('dqn_custom_fixed_angle_test/final_model.zip', models)
            self.assertFalse(any('smoke' in name for name in models))


if __name__ == '__main__':
    unittest.main()
