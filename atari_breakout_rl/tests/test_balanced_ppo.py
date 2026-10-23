"""Check update equivalence, robust loss, migration and resumable checkpoints."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from balanced_ppo import BalancedPPO, migrate_to_balanced, parameter_groups, value_error_loss
from env import environment_spec, make_env
from model_io import load_model


class BalancedPPOTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mse_global_mode_matches_sb3_update(self):
        env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
        try:
            source = PPO('MlpPolicy', env, seed=45, n_steps=16, batch_size=16, n_epochs=1,
                         policy_kwargs=dict(share_features_extractor=False), device='cpu')
            source.learn(16)
            with tempfile.TemporaryDirectory() as directory:
                source.set_logger(configure(directory, []))
                target = migrate_to_balanced(source, env, seed=45, n_steps=16, batch_size=16,
                    n_epochs=1, value_loss_type='mse', separate_grad_clip=False, device='cpu')
                target.set_logger(configure(directory, []))
                target.policy.optimizer.load_state_dict(copy.deepcopy(source.policy.optimizer.state_dict()))
                target.rollout_buffer = copy.deepcopy(source.rollout_buffer)
                target._current_progress_remaining = source._current_progress_remaining
                np.random.seed(17)
                source.train()
                np.random.seed(17)
                target.train()
                for key, value in source.policy.state_dict().items():
                    torch.testing.assert_close(value, target.policy.state_dict()[key], rtol=1e-5, atol=1e-7)
        finally:
            env.close()

    def test_huber_matches_small_errors_and_limits_outlier_gradient(self):
        values = torch.tensor([.5, -2., 8.], requires_grad=True)
        returns = torch.zeros_like(values)
        torch.testing.assert_close(value_error_loss(values, returns, 'huber', 10),
                                   value_error_loss(values, returns, 'mse'))
        outlier = torch.tensor([1000.], requires_grad=True)
        value_error_loss(outlier, torch.zeros_like(outlier), 'huber', 10).backward()
        self.assertAlmostEqual(float(outlier.grad), 20.)

    def test_critic_gradient_cannot_consume_actor_clip_budget(self):
        env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
        try:
            model = BalancedPPO('MlpPolicy', env, n_steps=16, batch_size=16,
                                n_epochs=1, device='cpu')
            model.learn(16)
            actor, critic = parameter_groups(model.policy)
            # Make the actual critic backward pass much larger than the actor,
            # then inspect gradients at the real optimizer update boundary.
            hooks = [p.register_hook(lambda grad: grad * 1000.) for p in actor]
            hooks += [p.register_hook(lambda grad: grad * 1000000.) for p in critic]
            captured = []
            def inspect_step(optimizer, args, kwargs):
                captured.append((float(torch.sqrt(sum(p.grad.square().sum() for p in actor if p.grad is not None))),
                                 float(torch.sqrt(sum(p.grad.square().sum() for p in critic if p.grad is not None)))))
            hooks.append(model.policy.optimizer.register_step_pre_hook(inspect_step))
            try:
                model.train()
            finally:
                for hook in hooks:
                    hook.remove()
            self.assertEqual(len(captured), 1)
            self.assertAlmostEqual(captured[0][0], .5, places=5)
            self.assertAlmostEqual(captured[0][1], 1., places=5)
        finally:
            env.close()

    def test_migration_cnn_load_and_continued_updates(self):
        env = make_env(42)
        try:
            source = PPO('CnnPolicy', env, n_steps=8, batch_size=8, device='cpu')
            source.num_timesteps = 400000
            model = migrate_to_balanced(source, env, seed=45, n_steps=8, batch_size=8,
                                         n_epochs=1, device='cpu')
            obs = env.reset()
            torch_obs, _ = source.policy.obs_to_tensor(obs)
            with torch.no_grad():
                torch.testing.assert_close(source.policy.get_distribution(torch_obs).distribution.probs,
                                           model.policy.get_distribution(torch_obs).distribution.probs)
                torch.testing.assert_close(source.policy.predict_values(torch_obs), model.policy.predict_values(torch_obs))
            model.breakout_algorithm = 'ppo'
            model.breakout_env_spec = environment_spec()
            with tempfile.TemporaryDirectory() as directory:
                model.learn(8)
                path = Path(directory) / 'balanced.zip'
                model.save(path)
                loaded = load_model(path, env=env, device='cpu')
                self.assertIsInstance(loaded, BalancedPPO)
                np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                              loaded.predict(obs, deterministic=True)[0])
                before = loaded._n_updates
                loaded.learn(8, reset_num_timesteps=False)
                self.assertGreater(loaded._n_updates, before)
                self.assertEqual(loaded.num_timesteps, 16)
                self.assertTrue(np.isfinite(loaded.logger.name_to_value['train/actor_grad_norm']))
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
