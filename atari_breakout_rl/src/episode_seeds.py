"""Explicit episode seeds for vector evaluation, independent of prior RNG use."""
import gymnasium as gym


class EpisodeSeeds(gym.Wrapper):
    def __init__(self, env, stride=1):
        super().__init__(env)
        if stride < 1:
            raise ValueError('Episode seed stride must be positive')
        self.stride = stride
        self.first_seed = None
        self.index = 0
        self.current_seed = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.first_seed, self.index = int(seed), 0
        if self.first_seed is None:
            raise ValueError('Seed the first evaluation reset explicitly')
        self.current_seed = self.first_seed + self.index * self.stride
        self.index += 1
        self.env.action_space.seed(self.current_seed)
        obs, info = self.env.reset(seed=self.current_seed, options=options)
        return obs, dict(info, episode_seed=self.current_seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, dict(info, episode_seed=self.current_seed)
