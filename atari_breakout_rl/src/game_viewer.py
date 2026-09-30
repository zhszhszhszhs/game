"""Live ALE playback: native intermediate frames, CPU inference, no recording."""
import io
from pathlib import Path
import threading
import time


class GameViewer:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.control_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.worker = None
        self.frame = self.jpeg = None
        self.frame_id = 0
        self.speed = 1.0
        self.state = self._initial_state()

    def _initial_state(self, agent='random', seed=20000, status='idle'):
        return dict(status=status, episode=0, reward=0, length=0, lives=None,
                    action='—', agent=agent, seed=seed, episode_seed=seed, error=None,
                    last_reward=None, history=[], game_seconds=0, auto_fire_count=0)

    def models(self):
        paths = sorted(self.model_dir.rglob('*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
        return [str(p.relative_to(self.model_dir)) for p in paths
                if p.resolve().is_relative_to(self.model_dir.resolve())]

    def snapshot(self):
        with self.lock:
            state = dict(self.state, frame_ready=self.frame is not None,
                         frame_id=self.frame_id, speed=self.speed)
            state['history'] = list(self.state['history'])
            if state['status'] in ('playing', 'paused'):
                state['status'] = 'paused' if self.pause_event.is_set() else 'playing'
        state['models'] = self.models()
        preferred = 'dqn_revised_500k/best_model/best_model.zip'
        state['recommended_model'] = preferred if preferred in state['models'] else next(
            (m for m in state['models'] if m.endswith('best_model.zip')), 'random')
        return state

    def image(self):
        with self.lock:
            return self.frame

    def stream_frame(self, after, timeout=2):
        with self.condition:
            self.condition.wait_for(lambda: self.frame_id != after and self.jpeg is not None,
                                    timeout=timeout)
            return self.frame_id, self.jpeg

    def set_speed(self, speed):
        if isinstance(speed, bool) or speed not in (0.5, 1, 2):
            raise ValueError('播放速度可选 0.5、1 或 2')
        with self.lock:
            self.speed = float(speed)

    def start(self, agent='random', seed=20000):
        if agent != 'random' and agent not in self.models():
            raise ValueError('请选择列表中的模型')
        if not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ValueError('Seed 必须是 0 到 2147483647 之间的整数')
        with self.control_lock:
            self._stop()
            self.stop_event.clear()
            self.pause_event.clear()
            with self.lock:
                self.frame = self.jpeg = None
                self.state = self._initial_state(agent, seed, 'loading')
            self.worker = threading.Thread(target=self._play, args=(agent, seed), daemon=True)
            self.worker.start()

    def _stop(self):
        self.stop_event.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=15)
            if self.worker.is_alive():
                raise RuntimeError('模型仍在加载，请稍后重试')

    def stop(self):
        with self.control_lock:
            self._stop()
            with self.lock:
                self.state['status'] = 'stopped'

    def pause(self, paused):
        if paused:
            self.pause_event.set()
        else:
            self.pause_event.clear()
        with self.lock:
            if self.state['status'] in ('playing', 'paused'):
                self.state['status'] = 'paused' if paused else 'playing'

    def _publish(self, frame, **updates):
        from PIL import Image
        picture = Image.fromarray(frame)
        png, jpeg = io.BytesIO(), io.BytesIO()
        picture.save(png, format='PNG')
        picture.save(jpeg, format='JPEG', quality=95, subsampling=0)
        with self.condition:
            self.frame, self.jpeg = png.getvalue(), jpeg.getvalue()
            self.frame_id += 1
            self.state.update(updates)
            self.condition.notify_all()

    def _ready(self):
        while self.pause_event.is_set() and not self.stop_event.is_set():
            self.stop_event.wait(.02)
        return not self.stop_event.is_set()

    def _play(self, agent, seed):
        env = None
        try:
            from stable_baselines3 import DQN
            from env import make_env
            from utils import setup
            setup(seed, 'cpu', threads=1)
            frames, capturing = [], False

            def capture(frame, reward, lives):
                nonlocal capturing
                if frame is None:
                    capturing = False
                elif capturing:
                    frames.append((frame.copy(), reward, lives))

            env = make_env(seed, render_mode='rgb_array', frame_callback=capture)
            model = None if agent == 'random' else DQN.load(self.model_dir / agent, env=env, device='cpu')
            meanings = env.env_method('get_action_meanings')[0]
            episode, last_reward, history = 0, None, []
            while not self.stop_event.is_set():
                episode_seed = seed + episode
                env.seed(episode_seed)
                env.action_space.seed(episode_seed)
                observation = env.reset()
                episode += 1
                total, length, raw_frames, auto_fires = 0.0, 0, 0, 0
                self._publish(env.get_images()[0], status='playing', episode=episode,
                              episode_seed=episode_seed, reward=0, length=0, action='—',
                              last_reward=last_reward, game_seconds=0, auto_fire_count=0,
                              lives=env.venv.venv.envs[0].unwrapped.ale.lives())
                while self._ready():
                    started = time.monotonic()
                    action = [env.action_space.sample()] if model is None else model.predict(observation, deterministic=True)[0]
                    frames.clear()
                    capturing = True
                    observation, rewards, dones, infos = env.step(action)
                    capturing = False
                    length += 1
                    label = meanings[int(action[0])]
                    if infos[0].get('ale_auto_fire'):
                        label += ' + 自动发球'
                        auto_fires += 1
                    displayed_reward = total
                    total += float(rewards[0])
                    # Show the real intermediate ALE frames skipped by the policy.
                    # A slow client sees the latest frame; no unbounded frame queue.
                    for frame, reward, lives in frames:
                        if not self._ready():
                            break
                        displayed_reward += reward
                        raw_frames += 1
                        self._publish(frame, reward=displayed_reward, length=length,
                                      action=label, lives=lives, status='playing',
                                      game_seconds=round(raw_frames / 60, 2), auto_fire_count=auto_fires)
                        with self.lock:
                            interval = 1 / (60 * self.speed)
                        self.stop_event.wait(max(0, interval - (time.monotonic() - started)))
                        started = time.monotonic()
                    if self.stop_event.is_set():
                        break
                    if dones[0]:
                        last_reward = total
                        history = (history + [dict(episode=episode, reward=total, length=length,
                                                  seed=episode_seed, truncated=bool(infos[0].get('TimeLimit.truncated', False)))])[-10:]
                        with self.lock:
                            self.state.update(status='episode_end', last_reward=last_reward,
                                              reward=total, history=history)
                        # Preserve the actual terminal frame before the next game.
                        self.stop_event.wait(1.5)
                        break
        except Exception as error:
            with self.lock:
                self.state.update(status='error', error=str(error))
        finally:
            if env is not None:
                env.close()
