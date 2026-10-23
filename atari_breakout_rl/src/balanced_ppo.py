"""PPO with independent gradient clipping and a robust value loss.

The update follows Stable-Baselines3 2.9.0 PPO (MIT license; see SB3_LICENSE.txt).
Sampling, GAE, distributions, optimizers and serialization remain in SB3.
"""
import math

import numpy as np
import torch
from gymnasium import spaces
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


def parameter_groups(policy):
    """Partition independent actor/critic parameters, rejecting shared weights."""
    actor = (list(policy.pi_features_extractor.parameters())
             + list(policy.mlp_extractor.policy_net.parameters())
             + list(policy.action_net.parameters()))
    if hasattr(policy, 'log_std'):
        actor.append(policy.log_std)
    critic = (list(policy.vf_features_extractor.parameters())
              + list(policy.mlp_extractor.value_net.parameters())
              + list(policy.value_net.parameters()))
    actor = list({id(p): p for p in actor if p.requires_grad}.values())
    critic = list({id(p): p for p in critic if p.requires_grad}.values())
    actor_ids, critic_ids = {id(p) for p in actor}, {id(p) for p in critic}
    if actor_ids & critic_ids:
        raise ValueError('Balanced PPO requires independent actor/critic networks')
    if actor_ids | critic_ids != {id(p) for p in policy.parameters() if p.requires_grad}:
        raise ValueError('Unassigned trainable policy parameters')
    return actor, critic


def value_error_loss(values, returns, kind='huber', delta=10.0):
    if kind == 'mse':
        return F.mse_loss(values, returns)
    if kind == 'huber':
        # Match MSE exactly for |error| <= delta; cap outlier derivatives.
        return 2.0 * F.huber_loss(values, returns, delta=delta)
    raise ValueError(f'Unknown value loss: {kind}')


class BalancedPPO(PPO):
    def __init__(self, *args, critic_max_grad_norm=1.0, value_loss_type='huber',
                 huber_delta=10.0, separate_grad_clip=True, **kwargs):
        if not math.isfinite(critic_max_grad_norm) or critic_max_grad_norm <= 0:
            raise ValueError('critic_max_grad_norm must be finite and positive')
        if not math.isfinite(huber_delta) or huber_delta <= 0 or value_loss_type not in ('mse', 'huber'):
            raise ValueError('Invalid value loss or Huber delta')
        self.critic_max_grad_norm = critic_max_grad_norm
        self.value_loss_type = value_loss_type
        self.huber_delta = huber_delta
        self.separate_grad_clip = separate_grad_clip
        kwargs['policy_kwargs'] = dict(kwargs.get('policy_kwargs') or {})
        kwargs['policy_kwargs'].setdefault('share_features_extractor', False)
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        parameter_groups(self.policy)
        self.breakout_ppo_variant = 'balanced'

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        self.logger.record('train/advantage_std', float(np.std(self.rollout_buffer.advantages)))
        self.logger.record('train/return_std', float(np.std(self.rollout_buffer.returns)))
        clip = self.clip_range(self._current_progress_remaining)
        value_clip = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf else None
        actor, critic = parameter_groups(self.policy)
        policy_losses, value_losses, entropy_losses, clip_fractions = [], [], [], []
        actor_norms, critic_norms, joint_scales, kl_values = [], [], [], []
        keep_training = True
        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.batch_size):
                actions = batch.actions.long().flatten() if isinstance(self.action_space, spaces.Discrete) else batch.actions
                values, log_prob, entropy = self.policy.evaluate_actions(batch.observations, actions)
                values = values.flatten()
                advantages = batch.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = torch.exp(log_prob - batch.old_log_prob)
                unclipped = advantages * ratio
                clipped = advantages * torch.clamp(ratio, 1 - clip, 1 + clip)
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                predicted = (batch.old_values + torch.clamp(values - batch.old_values, -value_clip, value_clip)
                             if value_clip is not None else values)
                value_loss = value_error_loss(predicted, batch.returns, self.value_loss_type, self.huber_delta)
                entropy_loss = -entropy.mean() if entropy is not None else log_prob.mean()
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite Balanced PPO loss')
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append((torch.abs(ratio - 1) > clip).float().mean().item())
                with torch.no_grad():
                    log_ratio = log_prob - batch.old_log_prob
                    kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                kl_values.append(kl)
                if self.target_kl is not None and kl > 1.5 * self.target_kl:
                    keep_training = False
                    if self.verbose:
                        print(f'KL guard stopped update: {kl:.5f}', flush=True)
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                if self.separate_grad_clip:
                    a = float(torch.nn.utils.clip_grad_norm_(actor, self.max_grad_norm, error_if_nonfinite=True))
                    c = float(torch.nn.utils.clip_grad_norm_(critic, self.critic_max_grad_norm, error_if_nonfinite=True))
                    actor_norms.append(a)
                    critic_norms.append(c)
                    joint_scales.append(min(1.0, self.max_grad_norm / (math.hypot(a, c) + 1e-6)))
                else:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm, error_if_nonfinite=True)
                self.policy.optimizer.step()
            self._n_updates += 1
            if not keep_training:
                break
        for key, values_to_log in [('policy_gradient_loss', policy_losses), ('value_loss', value_losses),
                                   ('entropy_loss', entropy_losses), ('clip_fraction', clip_fractions),
                                   ('approx_kl', kl_values), ('actor_grad_norm', actor_norms),
                                   ('critic_grad_norm', critic_norms), ('joint_clip_scale_if_used', joint_scales)]:
            if values_to_log:
                self.logger.record('train/' + key, float(np.mean(values_to_log)))
        self.logger.record('train/loss', loss.item())
        self.logger.record('train/clip_range', clip)
        if value_clip is not None:
            self.logger.record('train/clip_range_vf', value_clip)
        self.logger.record('train/explained_variance', explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()))
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        if hasattr(self.policy, 'log_std'):
            self.logger.record('train/std', torch.exp(self.policy.log_std).mean().item())
        self.log_policy_update(clip)
        self.log_value_fit('after_joint')

    def log_policy_update(self, clip):
        """Full-rollout final KL estimate; unlike averages over changing minibatches."""
        total_kl, total_clipped, count = 0.0, 0, 0
        size = self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs
        with torch.no_grad():
            for start in range(0, size, self.batch_size):
                batch = self.rollout_buffer._get_samples(np.arange(start, min(start + self.batch_size, size)))
                actions = batch.actions.long().flatten() if isinstance(self.action_space, spaces.Discrete) else batch.actions
                log_prob = self.policy.get_distribution(batch.observations).log_prob(actions)
                log_ratio = log_prob - batch.old_log_prob
                ratio = torch.exp(log_ratio)
                total_kl += ((ratio - 1) - log_ratio).sum().item()
                total_clipped += int((torch.abs(ratio - 1) > clip).sum().item())
                count += len(log_ratio)
        self.logger.record('train/policy_kl_after_update', total_kl / count)
        self.logger.record('train/policy_clip_fraction_after_update', total_clipped / count)
        return total_kl / count

    def log_value_fit(self, suffix):
        """Measure current critic fit without shuffling or changing rollout targets.

        The usual explained_variance uses values collected BEFORE optimization.
        These metrics evaluate the current network against the same GAE targets;
        they measure training fit, not independent value generalization.
        """
        predictions, targets = [], []
        size = self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs
        with torch.no_grad():
            for start in range(0, size, self.batch_size):
                batch = self.rollout_buffer._get_samples(np.arange(start, min(start + self.batch_size, size)))
                predictions.append(self.policy.predict_values(batch.observations).flatten().cpu().numpy())
                targets.append(batch.returns.cpu().numpy())
        values, returns = np.concatenate(predictions), np.concatenate(targets)
        errors = values - returns
        self.logger.record('train/value_mse_' + suffix, float(np.mean(errors ** 2)))
        self.logger.record('train/value_p90_error_' + suffix, float(np.percentile(np.abs(errors), 90)))
        self.logger.record('train/value_ev_' + suffix, explained_variance(values, returns))


def migrate_to_balanced(source, env, **kwargs):
    policy_kwargs = dict(source.policy_kwargs, share_features_extractor=False)
    model = BalancedPPO(source.policy_class, env, policy_kwargs=policy_kwargs, **kwargs)
    model.policy.load_state_dict(source.policy.state_dict(), strict=True)
    model.num_timesteps = source.num_timesteps
    model._n_updates = source._n_updates
    return model
