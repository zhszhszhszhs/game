"""SB3 replay persistence with optional streaming gzip compression."""
import gzip
import io
from pathlib import Path

import numpy as np
from stable_baselines3.common.buffers import NStepReplayBuffer
from env import environment_spec


def save_replay(model, path, compress=False):
    model.replay_buffer.breakout_env_spec = environment_spec(model)
    path = Path(path)
    if compress:
        path = path.with_suffix(path.suffix + '.gz')
    temporary = path.with_name(path.name + '.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compress else open
    options = {'compresslevel': 1} if compress else {}
    with opener(temporary, 'wb', **options) as stream:
        # SB3 accepts concrete BufferedWriter/Reader objects, not GzipFile itself.
        with io.BufferedWriter(stream) as buffered:
            model.save_replay_buffer(buffered)
    temporary.replace(path)
    return path


def load_replay(model, path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    previous = model.replay_buffer
    try:
        with opener(path, 'rb') as stream:
            with io.BufferedReader(stream) as buffered:
                model.load_replay_buffer(buffered)
        replay = model.replay_buffer
    finally:
        # Validation failures must leave the model's original replay usable.
        model.replay_buffer = previous
    actual_steps = replay.n_steps if isinstance(replay, NStepReplayBuffer) else 1
    if environment_spec(replay) != environment_spec(model):
        raise ValueError('Replay environment dynamics do not match the model; collect fresh replay')
    if actual_steps != model.n_steps:
        raise ValueError(f'Replay uses {actual_steps}-step returns, requested {model.n_steps}; '
                         'omit --replay-buffer and collect fresh experience when changing n_steps')
    if isinstance(replay, NStepReplayBuffer) and not np.isclose(replay.gamma, model.gamma):
        raise ValueError('Replay gamma does not match the requested model gamma')
    if (replay.buffer_size != model.buffer_size or replay.n_envs != model.n_envs
            or replay.obs_shape != model.observation_space.shape):
        raise ValueError('Replay capacity, environment count or observation shape does not match')
    if isinstance(replay, NStepReplayBuffer) and replay.size() > 0:
        # A restored game environment starts a new episode. End any unfinished
        # stored trajectory as a timeout so n-step returns cannot cross the reset;
        # bootstrap still uses the saved next observation of the old trajectory.
        last = (replay.pos - 1) % replay.buffer_size
        unfinished = replay.dones[last] == 0
        replay.timeouts[last, unfinished] = 1
        replay.dones[last, unfinished] = 1
    model.replay_buffer = replay
