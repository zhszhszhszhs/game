"""Bounded actor-only KL feedback over Balanced PPO's unchanged rollout update.

The controller observes the FINAL full-rollout sampled KL, smooths it, and
changes the NEXT rollout's actor learning rate. Critic LR, targets and Adam
moments are untouched. Disabling feedback gives the matched fixed-LR control.
"""
import math

from dual_rate_ppo import DualRatePPO


def validate_controller(target, band, factor, alpha, lower, upper):
    if not all(math.isfinite(x) for x in (target, band, factor, alpha, lower, upper)):
        raise ValueError('Adaptive KL settings must be finite')
    if not (target > 0 and band > 1 and factor > 1 and 0 < alpha <= 1
            and 0 < lower <= 1 <= upper):
        raise ValueError('Require target>0, band/factor>1, alpha in (0,1], 0<lower<=1<=upper')


class AdaptiveKLPPO(DualRatePPO):
    def __init__(self, *args, adaptive_kl_enabled=True, adaptive_kl_target=.0005,
                 adaptive_kl_band=1.5, adaptive_kl_factor=1.1, adaptive_kl_alpha=.2,
                 actor_lr_min_multiplier=.5, actor_lr_max_multiplier=2., **kwargs):
        validate_controller(adaptive_kl_target, adaptive_kl_band, adaptive_kl_factor,
                            adaptive_kl_alpha, actor_lr_min_multiplier, actor_lr_max_multiplier)
        if kwargs.pop('critic_lr_multiplier', 1.) != 1. or kwargs.pop('extra_critic_epochs', 0) != 0:
            raise ValueError('Adaptive KL requires critic_lr_multiplier=1 and extra_critic_epochs=0')
        self.adaptive_kl_enabled = adaptive_kl_enabled
        self.adaptive_kl_target = adaptive_kl_target
        self.adaptive_kl_band = adaptive_kl_band
        self.adaptive_kl_factor = adaptive_kl_factor
        self.adaptive_kl_alpha = adaptive_kl_alpha
        self.actor_lr_min_multiplier = actor_lr_min_multiplier
        self.actor_lr_max_multiplier = actor_lr_max_multiplier
        self.reset_controller()
        super().__init__(*args, critic_lr_multiplier=1., extra_critic_epochs=0, **kwargs)

    def reset_controller(self):
        self.actor_lr_multiplier = 1.
        self.kl_ema = None
        self.kl_controller_updates = 0

    def _setup_model(self):
        super()._setup_model()
        self.breakout_ppo_variant = 'adaptive_kl'

    def _update_learning_rate(self, optimizers):
        super()._update_learning_rate(optimizers)
        multiplier = self.actor_lr_multiplier if self.adaptive_kl_enabled else 1.
        optimizers.param_groups[0]['lr'] *= multiplier
        self.logger.record('train/actor_learning_rate', optimizers.param_groups[0]['lr'])
        self.logger.record('train/actor_lr_multiplier', multiplier)

    def observe_kl(self, kl):
        if not math.isfinite(kl) or kl < 0:
            raise FloatingPointError('Full-rollout KL must be finite and nonnegative')
        self.kl_ema = kl if self.kl_ema is None else (
            (1-self.adaptive_kl_alpha)*self.kl_ema + self.adaptive_kl_alpha*kl)
        self.kl_controller_updates += 1
        if self.adaptive_kl_enabled:
            # A sudden large update overrides the smoothing delay.
            if kl > self.adaptive_kl_target*self.adaptive_kl_band or (
                    self.kl_ema > self.adaptive_kl_target*self.adaptive_kl_band):
                self.actor_lr_multiplier /= self.adaptive_kl_factor
            elif self.kl_ema < self.adaptive_kl_target/self.adaptive_kl_band:
                self.actor_lr_multiplier *= self.adaptive_kl_factor
            self.actor_lr_multiplier = min(self.actor_lr_max_multiplier,
                max(self.actor_lr_min_multiplier, self.actor_lr_multiplier))
        else:
            self.actor_lr_multiplier = 1.

    def log_policy_update(self, clip):
        kl = super().log_policy_update(clip)
        self.observe_kl(kl)
        self.logger.record('train/kl_ema', self.kl_ema)
        self.logger.record('train/adaptive_kl_target', self.adaptive_kl_target)
        self.logger.record('train/actor_lr_multiplier_next', self.actor_lr_multiplier)
        return kl


def migrate_to_adaptive_kl(source, env, **kwargs):
    """Copy both CNNs exactly; create fresh partitioned Adam and controller."""
    model = AdaptiveKLPPO(source.policy_class, env,
        policy_kwargs=dict(source.policy_kwargs, share_features_extractor=False), **kwargs)
    model.policy.load_state_dict(source.policy.state_dict(), strict=True)
    model.num_timesteps = source.num_timesteps
    model._n_updates = source._n_updates
    return model
