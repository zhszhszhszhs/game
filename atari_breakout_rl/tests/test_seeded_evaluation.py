"""Guard per-episode seeds, callback accounting and bounded stable selection."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from callbacks import RunState, TrackedEvalCallback
from checkpoint_selection import choose_checkpoint
from env import make_env
from episode_seeds import EpisodeSeeds
from train_ppo import parse_args


class VariableDraws(gym.Env):
    observation_space = gym.spaces.Box(0, 1, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.value = float(self.np_random.random())
        return np.array([self.value, 0], np.float32), {}

    def step(self, action):
        # Different policies advance RNG by different amounts within a game.
        self.np_random.random(1 + int(action) * 20)
        return np.zeros(2, np.float32), self.value * 100., True, False, {}


class SeededEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_seeds_do_not_depend_on_previous_game_rng_consumption(self):
        left, right = EpisodeSeeds(VariableDraws(), 3), EpisodeSeeds(VariableDraws(), 3)
        try:
            for episode in range(4):
                a, info_a = left.reset(seed=100 if episode == 0 else None)
                b, info_b = right.reset(seed=100 if episode == 0 else None)
                np.testing.assert_array_equal(a, b)
                self.assertEqual(info_a['episode_seed'], 100 + 3 * episode)
                self.assertEqual(info_a, info_b)
                self.assertEqual(left.step(0)[-1]['episode_seed'], 100 + 3 * episode)
                right.step(1)
            a, _ = left.reset(seed=100)
            reference, _ = right.reset(seed=100)
            np.testing.assert_array_equal(a, reference)
        finally:
            left.close()
            right.close()

    def test_breakout_vector_autoreset_assigns_exact_seed_list(self):
        env = make_env(111000, n_envs=2, fixed_episode_seeds=True)
        try:
            original = env.reset()
            for wrapped in env.venv.venv.envs:
                wrapped.unwrapped.max_frames = 4
            for expected in [[111000, 111001], [111002, 111003]]:
                _, _, done, infos = env.step([0, 0])
                self.assertTrue(done.all())
                self.assertEqual([info['episode_seed'] for info in infos], expected)
                self.assertEqual([info['episode']['episode_seed'] for info in infos], expected)
                self.assertEqual(infos[0]['terminal_observation'].shape, (4, 84, 84))
            env.seed(111000)
            np.testing.assert_array_equal(env.reset(), original)
        finally:
            env.close()

    def test_callback_persists_seed_aligned_results_and_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = DummyVecEnv([lambda: Monitor(VariableDraws())])
            evaluate = DummyVecEnv([lambda: Monitor(EpisodeSeeds(VariableDraws(), 2)) for _ in range(2)])
            try:
                model = DQN('MlpPolicy', train, buffer_size=16, device='cpu', seed=47)
                model.set_logger(configure(str(root / 'logs'), []))
                callback = TrackedEvalCallback(evaluate, state=RunState(root, 8), eval_seed=101000,
                    fixed_episode_seeds=True, evaluate_at_start=True, n_eval_episodes=4,
                    eval_freq=1, log_path=str(root / 'eval'), verbose=0)
                callback.init_callback(model)
                callback.on_training_start({}, {})
                with (root / 'eval/episodes_0.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([int(row['seed']) for row in rows], list(range(101000, 101004)))
                for row in rows:
                    expected = np.random.default_rng(int(row['seed'])).random() * 100
                    self.assertAlmostEqual(float(row['reward']), expected, places=5)
                initial = callback.evaluations_results[-1]
                model.num_timesteps = callback.num_timesteps = callback.n_calls = 4
                callback._on_step()
                np.testing.assert_array_equal(callback.evaluations_results[-1], initial)
                self.assertEqual(len(callback.episode_rows), 4)
                self.assertTrue((root / 'eval/episodes_4.csv').exists())
            finally:
                train.close()
                evaluate.close()

    def test_evaluation_heartbeat_updates_before_an_episode_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            env = DummyVecEnv([lambda: Monitor(VariableDraws())])
            try:
                model = DQN('MlpPolicy', env, buffer_size=16, device='cpu')
                callback = TrackedEvalCallback(env, state=RunState(Path(directory), 8),
                    eval_seed=101000, n_eval_episodes=1, verbose=0)
                callback.init_callback(model)
                callback.last_eval_heartbeat = -100
                callback._log_success_callback({'done':False, 'info':{}}, {})
                state = json.loads((Path(directory) / 'runtime.json').read_text())
                self.assertEqual(state['status'], 'evaluating')
                self.assertEqual(state['evaluation_episodes_completed'], 0)
                self.assertEqual(state['evaluation_transitions'], 1)
            finally:
                env.close()

    def test_stable_choice_cannot_exceed_the_mean_drop_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind in ['best_model', 'stable_model']:
                (root / kind).mkdir()
                (root / kind / 'best_model.zip').touch()
            (root / 'best_model/selection.json').write_text(json.dumps(dict(mean_reward=100, median_reward=50)))
            for mean, median, expected in [(99, 80, 'stable_model'), (94, 90, 'best_model'),
                                            (95, 70, 'stable_model'), (99, 40, 'best_model')]:
                with self.subTest(mean=mean, median=median):
                    (root / 'stable_model/selection.json').write_text(json.dumps(dict(mean_reward=mean, median_reward=median)))
                    selected, metadata = choose_checkpoint(root, 'stable_near_mean', .05)
                    self.assertEqual(selected.parent.name, expected)
                    self.assertEqual(metadata['minimum_mean_reward'], 95)
            selected, _ = choose_checkpoint(root, 'mean')
            self.assertEqual(selected.parent.name, 'best_model')

    def test_invalid_fixed_seed_and_selection_options_rejected(self):
        for flags in [['--fixed-eval-seeds', '--eval-envs', '2', '--n-eval-episodes', '3'],
                      ['--selection-mean-tolerance', 'nan'], ['--selection-mean-tolerance', '1']]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(flags)


if __name__ == '__main__':
    unittest.main()
