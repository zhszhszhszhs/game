"""Load tagged PPO models and existing DQN checkpoints through one interface."""
import json
from zipfile import ZipFile

from stable_baselines3 import DQN, PPO
from env import environment_spec


def load_model(path, **kwargs):
    with ZipFile(path) as archive:
        data = json.loads(archive.read('data'))
    algorithm = data.get('breakout_algorithm', 'dqn')
    if algorithm not in ('dqn', 'ppo'):
        raise ValueError(f'Unsupported algorithm: {algorithm}')
    variant = data.get('breakout_ppo_variant', 'standard')
    if algorithm == 'ppo' and variant == 'balanced':
        from balanced_ppo import BalancedPPO
        model_class = BalancedPPO
    elif algorithm == 'ppo' and variant == 'dual_rate':
        from dual_rate_ppo import DualRatePPO
        model_class = DualRatePPO
    elif algorithm == 'ppo' and variant == 'adaptive_kl':
        from adaptive_kl_ppo import AdaptiveKLPPO
        model_class = AdaptiveKLPPO
    elif algorithm == 'ppo' and variant != 'standard':
        raise ValueError(f'Unsupported PPO variant: {variant}')
    else:
        model_class = {'dqn': DQN, 'ppo': PPO}[algorithm]
    model = model_class.load(path, **kwargs)
    environment_spec(model)
    return model


def algorithm_name(model):
    return 'ppo' if isinstance(model, PPO) else 'dqn'
