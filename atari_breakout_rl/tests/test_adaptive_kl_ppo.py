"""Check feedback isolation, neutral equivalence and saved controller/Adam state."""
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
from adaptive_kl_ppo import AdaptiveKLPPO, migrate_to_adaptive_kl
from balanced_ppo import BalancedPPO, parameter_groups
from env import environment_spec, make_env
from model_io import load_model
from train_ppo import parse_args
from run_ppo_study import paired_summary


class AdaptiveKLPPOTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_disabled_feedback_matches_balanced_update(self):
        env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
        try:
            source = BalancedPPO('MlpPolicy', env, seed=48, n_steps=16, batch_size=8,
                                 n_epochs=2, device='cpu')
            source.learn(16)
            target = migrate_to_adaptive_kl(source, env, seed=48, n_steps=16, batch_size=8,
                n_epochs=2, adaptive_kl_enabled=False, device='cpu')
            source.policy.optimizer = source.policy.optimizer_class(source.policy.parameters(),
                lr=source.lr_schedule(1.), **source.policy.optimizer_kwargs)
            with tempfile.TemporaryDirectory() as directory:
                for model in (source, target):model.set_logger(configure(directory, []))
                target.rollout_buffer = copy.deepcopy(source.rollout_buffer)
                target._current_progress_remaining = source._current_progress_remaining
                np.random.seed(17); source.train()
                np.random.seed(17); target.train()
                for key, value in source.policy.state_dict().items():
                    torch.testing.assert_close(value, target.policy.state_dict()[key], rtol=1e-5, atol=1e-7)
                self.assertEqual(target.actor_lr_multiplier, 1.)
        finally:
            env.close()

    def test_feedback_bounds_smoothing_and_critic_lr_isolation(self):
        env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
        try:
            model = AdaptiveKLPPO('MlpPolicy', env, n_steps=8, batch_size=8, device='cpu',
                learning_rate=LinearSchedule(2e-5, 5e-6, 1.), adaptive_kl_alpha=.2)
            with tempfile.TemporaryDirectory() as directory:
                model.set_logger(configure(directory, []))
                model._current_progress_remaining = .5
                for _ in range(100):model.observe_kl(0.)
                self.assertEqual(model.actor_lr_multiplier, 2.)
                model._update_learning_rate(model.policy.optimizer)
                actor, critic = model.policy.optimizer.param_groups
                self.assertAlmostEqual(actor['lr'], 2.5e-5)
                self.assertAlmostEqual(critic['lr'], 1.25e-5)
                model.reset_controller()
                model.observe_kl(.0005)
                self.assertEqual(model.actor_lr_multiplier, 1.)
                model.observe_kl(.001)
                self.assertAlmostEqual(model.kl_ema, .0006)
                self.assertLess(model.actor_lr_multiplier, 1.)  # Immediate spike response.
                for _ in range(100):model.observe_kl(.01)
                self.assertEqual(model.actor_lr_multiplier, .5)
                for invalid in [float('nan'),float('inf'),-1.]:
                    with self.assertRaises(FloatingPointError):model.observe_kl(invalid)
        finally:
            env.close()

    def test_existing_kl_guard_still_freezes_policy(self):
        env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
        try:
            model = AdaptiveKLPPO('MlpPolicy', env, n_steps=16, batch_size=8, n_epochs=1,
                                  target_kl=.001, seed=48, device='cpu')
            model.learn(16)
            actor, _ = parameter_groups(model.policy)
            before = [p.detach().clone() for p in actor]
            before_multiplier = model.actor_lr_multiplier
            model.rollout_buffer.log_probs += 10.
            model.train()
            for param, value in zip(actor,before):torch.testing.assert_close(param,value,rtol=0,atol=0)
            self.assertLess(model.actor_lr_multiplier, before_multiplier)
        finally:
            env.close()

    def test_cnn_migration_save_load_and_next_update_match(self):
        env = make_env(48)
        try:
            source = BalancedPPO('CnnPolicy',env,n_steps=8,batch_size=8,n_epochs=1,device='cpu')
            model = migrate_to_adaptive_kl(source,env,n_steps=8,batch_size=8,n_epochs=1,device='cpu')
            obs,_ = model.policy.obs_to_tensor(env.reset())
            with torch.no_grad():
                torch.testing.assert_close(source.policy.get_distribution(obs).distribution.probs,
                                           model.policy.get_distribution(obs).distribution.probs,rtol=0,atol=0)
                torch.testing.assert_close(source.policy.predict_values(obs),model.policy.predict_values(obs),rtol=0,atol=0)
            model.breakout_algorithm='ppo';model.breakout_env_spec=environment_spec()
            with tempfile.TemporaryDirectory() as directory:
                model.set_logger(configure(directory,[]))
                model.learn(8)
                model.observe_kl(.004)
                path=Path(directory)/'adaptive.zip';model.save(path)
                loaded=load_model(path,env=env,device='cpu')
                self.assertIsInstance(loaded,AdaptiveKLPPO)
                self.assertEqual(model.kl_ema,loaded.kl_ema)
                self.assertEqual(model.actor_lr_multiplier,loaded.actor_lr_multiplier)
                self.assertEqual(model.kl_controller_updates,loaded.kl_controller_updates)
                for p,q in zip(model.policy.parameters(),loaded.policy.parameters()):
                    for key,value in model.policy.optimizer.state[p].items():
                        torch.testing.assert_close(value,loaded.policy.optimizer.state[q][key],rtol=0,atol=0)
                loaded.set_logger(configure(directory,[]))
                loaded.rollout_buffer=copy.deepcopy(model.rollout_buffer)
                loaded._current_progress_remaining=model._current_progress_remaining
                np.random.seed(17);model.train()
                np.random.seed(17);loaded.train()
                for key,value in model.policy.state_dict().items():
                    torch.testing.assert_close(value,loaded.policy.state_dict()[key],rtol=0,atol=0)
                self.assertEqual(model.actor_lr_multiplier,loaded.actor_lr_multiplier)
        finally:
            env.close()

    def test_invalid_controller_and_confounded_settings_rejected(self):
        for flags in [['--adaptive-kl-target','nan'],['--adaptive-kl-band','1'],
                      ['--adaptive-kl-alpha','0'],['--adaptive-kl-factor','1'],
                      ['--actor-lr-min-multiplier','1.5'],['--actor-lr-max-multiplier','.9'],
                      ['--ppo-variant','adaptive_kl']]:
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parse_args(flags)

    def test_paired_summary_resamples_training_and_game_seed_pairs(self):
        control=np.arange(150,dtype=float).reshape(3,50)
        result=paired_summary(control+7,control)
        self.assertEqual(result['mean_difference'],7.)
        self.assertEqual(result['ci95'],[7.,7.])
        self.assertEqual(result['per_training_seed_mean_difference'],[7.,7.,7.])

    def test_paired_summary_rejects_missing_or_invalid_evaluations(self):
        for adaptive,control in [(np.zeros((3,49)),np.zeros((3,50))),
                                 (np.full((3,50),np.nan),np.zeros((3,50)))]:
            with self.assertRaises(ValueError):paired_summary(adaptive,control)


if __name__=='__main__':unittest.main()
