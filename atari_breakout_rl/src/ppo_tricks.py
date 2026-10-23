"""Optional PPO fine-tuning tools; inference stays compatible with standard SB3."""
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


def split_actor_critic(source, env, **kwargs):
    """Copy a CNN PPO into independent actor/critic extractors, with a fresh Adam.

    SB3 state_dict exposes pi/vf extractor aliases even for a shared CNN, so
    copying it initializes both independent networks to the inherited weights.
    The optimizer must be reset because its parameter groups have changed.
    """
    policy_kwargs = dict(source.policy_kwargs, share_features_extractor=False)
    model = PPO(source.policy_class, env, policy_kwargs=policy_kwargs, **kwargs)
    model.policy.load_state_dict(source.policy.state_dict(), strict=True)
    model.num_timesteps = source.num_timesteps
    model._n_updates = source._n_updates
    return model


class EntropySchedule(BaseCallback):
    """Linear entropy decay over newly requested interactions, including resumes."""
    def __init__(self, initial, final, timesteps):
        super().__init__()
        self.initial, self.final, self.timesteps = initial, final, timesteps

    def _on_training_start(self):
        self.start = self.model.num_timesteps
        self.model.ent_coef = self.initial

    def _on_rollout_start(self):
        fraction = min(1.0, max(0.0, (self.model.num_timesteps - self.start) / self.timesteps))
        self.model.ent_coef = self.initial + fraction * (self.final - self.initial)
        self.logger.record('train/ent_coef', self.model.ent_coef)

    def _on_step(self):
        return True
