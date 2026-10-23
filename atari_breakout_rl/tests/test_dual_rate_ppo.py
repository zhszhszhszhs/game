"""Check independent optimizer state, neutral equivalence and resumable CNNs."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from balanced_ppo import BalancedPPO, parameter_groups
from dual_rate_ppo import DualRatePPO, migrate_to_dual_rate
from env import environment_spec, make_env
from model_io import load_model
from train_ppo import parse_args


class DualRatePPOTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_neutral_variant_matches_balanced_update(self):
        env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
        try:
            source = BalancedPPO('MlpPolicy', env, seed=46, n_steps=16, batch_size=8,
                                 n_epochs=2, device='cpu')
            source.learn(16)
            with tempfile.TemporaryDirectory() as directory:
                target = migrate_to_dual_rate(source, env, seed=46, n_steps=16, batch_size=8,
                    n_epochs=2, critic_lr_multiplier=1., extra_critic_epochs=0, device='cpu')
                # Both sides start with the same weights and fresh Adam moments.
                source.policy.optimizer = source.policy.optimizer_class(source.policy.parameters(),
                    lr=source.lr_schedule(1.), **source.policy.optimizer_kwargs)
                for model in (source, target):
                    model.set_logger(configure(directory, []))
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

    def test_extra_critic_keeps_actor_weights_adam_and_rollout_fixed(self):
        env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
        try:
            model = DualRatePPO('MlpPolicy', env, n_steps=16, batch_size=8,
                n_epochs=1, extra_critic_epochs=2, seed=46, device='cpu')
            model.learn(16)
            actor, critic = parameter_groups(model.policy)
            actor_weights = [p.detach().clone() for p in actor]
            actor_state = [copy.deepcopy(model.policy.optimizer.state[p]) for p in actor]
            critic_weights = [p.detach().clone() for p in critic]
            frozen = {name: getattr(model.rollout_buffer, name).copy()
                      for name in ('returns', 'advantages', 'log_probs', 'values')}
            model._train_extra_critic()
            for param, weight, state in zip(actor, actor_weights, actor_state):
                torch.testing.assert_close(param, weight, rtol=0, atol=0)
                self.assertIsNone(param.grad)
                for key, value in state.items():
                    torch.testing.assert_close(model.policy.optimizer.state[param][key], value, rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(p, w) for p, w in zip(critic, critic_weights)))
            for name, value in frozen.items():
                np.testing.assert_array_equal(getattr(model.rollout_buffer, name), value)
            self.assertTrue(np.isfinite(model.logger.name_to_value['train/value_mse_after_extra']))
        finally:
            env.close()

    def test_kl_guard_freezes_actor_but_allows_extra_value_fitting(self):
        env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
        try:
            model = DualRatePPO('MlpPolicy', env, n_steps=16, batch_size=8, n_epochs=1,
                extra_critic_epochs=1, target_kl=.001, seed=46, device='cpu')
            model.learn(16)
            actor, critic = parameter_groups(model.policy)
            before_actor = [p.detach().clone() for p in actor]
            before_critic = [p.detach().clone() for p in critic]
            model.rollout_buffer.log_probs += 10.  # Guaranteed KL guard on the first minibatch.
            model.train()
            for p, value in zip(actor, before_actor):
                torch.testing.assert_close(p, value, rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(p, value) for p, value in zip(critic, before_critic)))
        finally:
            env.close()

    def test_cnn_migration_group_schedules_save_and_resume(self):
        env = make_env(46)
        try:
            source = BalancedPPO('CnnPolicy', env, n_steps=8, batch_size=8, n_epochs=1, device='cpu')
            model = migrate_to_dual_rate(source, env, n_steps=8, batch_size=8, n_epochs=1,
                learning_rate=LinearSchedule(2e-5, 5e-6, 1.), device='cpu')
            obs, _ = model.policy.obs_to_tensor(env.reset())
            with torch.no_grad():
                torch.testing.assert_close(source.policy.get_distribution(obs).distribution.probs,
                                           model.policy.get_distribution(obs).distribution.probs)
                torch.testing.assert_close(source.policy.predict_values(obs), model.policy.predict_values(obs))
            model.breakout_algorithm = 'ppo'
            model.breakout_env_spec = environment_spec()
            with tempfile.TemporaryDirectory() as directory:
                model.set_logger(configure(directory, []))
                model._current_progress_remaining = .5
                model._update_learning_rate(model.policy.optimizer)
                self.assertAlmostEqual(model.policy.optimizer.param_groups[0]['lr'], 1.25e-5)
                self.assertAlmostEqual(model.policy.optimizer.param_groups[1]['lr'], 3.75e-5)
                model.learn(8)
                path = Path(directory) / 'dual_rate.zip'
                model.save(path)
                loaded = load_model(path, env=env, device='cpu')
                self.assertIsInstance(loaded, DualRatePPO)
                self.assertEqual([g['role'] for g in loaded.policy.optimizer.param_groups], ['actor', 'critic'])
                for p, q in zip(model.policy.parameters(), loaded.policy.parameters()):
                    for key, value in model.policy.optimizer.state[p].items():
                        torch.testing.assert_close(value, loaded.policy.optimizer.state[q][key])
                loaded.set_logger(configure(directory, []))
                loaded.rollout_buffer = copy.deepcopy(model.rollout_buffer)
                loaded._current_progress_remaining = model._current_progress_remaining
                np.random.seed(17)
                model.train()
                np.random.seed(17)
                loaded.train()
                for key, value in model.policy.state_dict().items():
                    torch.testing.assert_close(value, loaded.policy.state_dict()[key], rtol=0, atol=0)
                loaded.learn(8, reset_num_timesteps=False)
                self.assertEqual(loaded.num_timesteps, 16)
        finally:
            env.close()

    def test_invalid_critic_settings_rejected(self):
        for flags in [['--critic-lr-multiplier', 'nan'], ['--critic-lr-multiplier', '0'],
                      ['--extra-critic-epochs', '-1']]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(flags)


if __name__ == '__main__':
    unittest.main()
