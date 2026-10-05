"""Protect the inherited policy and separate outlier-driven means from typical scores."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from callbacks import RunState, TrackedEvalCallback
from env import environment_spec
from replay_io import load_replay


class ScoreSequence(gym.Env):
    observation_space = gym.spaces.Box(0, 1, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Discrete(1)

    def __init__(self):
        self.scores = [10, 10, 10, 10]
        self.index = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, np.float32), {}

    def step(self, action):
        reward = self.scores[self.index % len(self.scores)]
        self.index += 1
        return np.zeros(2, np.float32), reward, True, False, {}


class CheckpointSelection(unittest.TestCase):
    def test_warmup_does_not_consume_early_stop_patience(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_env = DummyVecEnv([lambda: Monitor(ScoreSequence())])
            eval_env = DummyVecEnv([lambda: Monitor(ScoreSequence())])
            try:
                model = DQN('MlpPolicy', train_env, buffer_size=64,
                            learning_starts=8, batch_size=4, device='cpu')
                model.set_logger(configure(str(root / 'logs'), []))
                stopper = StopTrainingOnNoModelImprovement(1, min_evals=1)
                callback = TrackedEvalCallback(eval_env, state=RunState(root, 24),
                    eval_seed=31000, evaluate_at_start=True, eval_start_step=9,
                    callback_after_eval=stopper, log_path=str(root / 'eval'),
                    n_eval_episodes=4, eval_freq=4, verbose=0)
                model.learn(24, callback=callback)
                np.testing.assert_array_equal(callback.evaluations_timesteps, [0, 12, 16])
                self.assertEqual(model.num_timesteps, 16)
                self.assertGreater(model._n_updates, 0)
                self.assertEqual(stopper.no_improvement_evals, 2)
            finally:
                train_env.close()
                eval_env.close()

    def test_best_replay_is_loadable_and_not_replaced_by_worse_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = ScoreSequence()
            env = DummyVecEnv([lambda: Monitor(source)])
            try:
                model = DQN('MlpPolicy', env, buffer_size=16,
                            learning_starts=100, device='cpu')
                model.breakout_env_spec = environment_spec()
                model.learn(8)
                model.set_logger(configure(str(root / 'logs'), []))
                callback = TrackedEvalCallback(env, state=RunState(root, 20),
                    eval_seed=31000, evaluate_at_start=True, save_best_replay=True,
                    compress_replay=True, best_model_save_path=str(root / 'best'),
                    log_path=str(root / 'eval'), n_eval_episodes=4, eval_freq=4, verbose=0)
                callback.init_callback(model)
                callback.on_training_start({}, {})
                path = root / 'best/replay_buffer.pkl.gz'
                original = path.read_bytes()
                saved = DQN.load(root / 'best/best_model.zip', env=env, device='cpu')
                load_replay(saved, path)
                self.assertEqual(saved.replay_buffer.size(), 8)
                np.testing.assert_array_equal(saved.replay_buffer.rewards[:8],
                                              model.replay_buffer.rewards[:8])
                source.scores = [1, 1, 1, 1]
                model.num_timesteps = callback.num_timesteps = callback.n_calls = 12
                callback._on_step()
                self.assertEqual(path.read_bytes(), original)
                selection = json.loads((root / 'best/selection.json').read_text())
                self.assertEqual(selection['replay_size'], 8)
                self.assertEqual(selection['replay_buffer_path'], str(path))
            finally:
                env.close()

    def test_early_stop_counts_evaluations_and_preserves_starting_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_env = DummyVecEnv([lambda: Monitor(ScoreSequence())])
            eval_env = DummyVecEnv([lambda: Monitor(ScoreSequence())])
            try:
                model = DQN('MlpPolicy', train_env, buffer_size=16,
                            learning_starts=100, device='cpu')
                model.set_logger(configure(str(root / 'logs'), []))
                stopper = StopTrainingOnNoModelImprovement(1, min_evals=1)
                callback = TrackedEvalCallback(eval_env, state=RunState(root, 20),
                    eval_seed=31000, evaluate_at_start=True, callback_after_eval=stopper,
                    best_model_save_path=str(root / 'best'), log_path=str(root / 'eval'),
                    n_eval_episodes=4, eval_freq=4, verbose=0)
                model.learn(20, callback=callback)
                self.assertEqual(model.num_timesteps, 8)
                self.assertEqual(stopper.no_improvement_evals, 2)
                self.assertEqual(DQN.load(root / 'best/best_model.zip').num_timesteps, 0)
                selection = json.loads((root / 'best/selection.json').read_text())
                self.assertEqual(selection['timesteps'], 0)
                self.assertEqual(selection['mean_reward'], 10)
                np.testing.assert_array_equal(callback.evaluations_timesteps, [0, 4, 8])
            finally:
                train_env.close()
                eval_env.close()

    def test_initial_policy_and_separate_stability_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = ScoreSequence()
            env = DummyVecEnv([lambda: Monitor(source)])
            try:
                model = DQN('MlpPolicy', env, buffer_size=16, device='cpu')
                model.set_logger(configure(str(root / 'logs'), []))
                callback = TrackedEvalCallback(env, state=RunState(root, 8), eval_seed=31000,
                    evaluate_at_start=True, stable_model_save_path=root / 'stable',
                    best_model_save_path=str(root / 'mean'), log_path=str(root / 'eval'),
                    n_eval_episodes=4, eval_freq=4, verbose=0)
                callback.init_callback(model)
                callback.on_training_start({}, {})
                self.assertEqual(callback.best_mean_reward, 10)
                self.assertEqual(callback.best_stability, (10, 10))
                self.assertTrue((root / 'mean/best_model.zip').exists())

                source.scores = [0, 0, 0, 100]
                model.num_timesteps = callback.num_timesteps = callback.n_calls = 4
                callback._on_step()
                self.assertEqual(callback.best_mean_reward, 25)
                self.assertEqual(callback.best_stability, (10, 10))
                selection = json.loads((root / 'stable/selection.json').read_text())
                self.assertEqual(selection['timesteps'], 0)

                source.scores = [15, 15, 15, 15]
                model.num_timesteps = callback.num_timesteps = callback.n_calls = 8
                callback._on_step()
                self.assertEqual(callback.best_mean_reward, 25)
                self.assertEqual(callback.best_stability, (15, 15))
                mean_model = DQN.load(root / 'mean/best_model.zip', device='cpu')
                stable_model = DQN.load(root / 'stable/best_model.zip', device='cpu')
                self.assertEqual(mean_model.num_timesteps, 4)
                self.assertEqual(stable_model.num_timesteps, 8)
                np.testing.assert_array_equal(callback.evaluations_timesteps, [0, 4, 8])
            finally:
                env.close()


if __name__ == '__main__':
    unittest.main()
