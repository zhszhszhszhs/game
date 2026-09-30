"""Live ALE gameplay for the web UI. CPU inference, no training or recording."""
import io
from pathlib import Path
import threading
import time


class GameViewer:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.lock = threading.Lock()
        self.control_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.worker = None
        self.frame = None
        self.state = dict(status='idle', episode=0, reward=0, length=0, lives=None,
                          action='—', agent='random', error=None, last_reward=None)

    def models(self):
        paths = sorted(self.model_dir.rglob('*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
        return [str(p.relative_to(self.model_dir)) for p in paths if p.resolve().is_relative_to(self.model_dir.resolve())]

    def snapshot(self):
        with self.lock:
            state = dict(self.state, frame_ready=self.frame is not None)
            if state['status'] in ('playing', 'paused'):
                state['status'] = 'paused' if self.pause_event.is_set() else 'playing'
        state['models'] = self.models()
        return state

    def image(self):
        with self.lock:
            return self.frame

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
                self.frame = None
                self.state = dict(status='loading', episode=0, reward=0, length=0,
                    lives=None, action='—', agent=agent, seed=seed, error=None, last_reward=None)
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
        stream = io.BytesIO()
        Image.fromarray(frame).save(stream, format='PNG')
        with self.lock:
            self.frame = stream.getvalue()
            self.state.update(updates)

    def _play(self, agent, seed):
        env = None
        try:
            # Lazy imports keep the page lightweight until the user starts a game.
            from stable_baselines3 import DQN
            from env import make_env
            from utils import setup
            setup(seed, 'cpu', threads=1)
            env = make_env(seed, render_mode='rgb_array')
            model = None if agent == 'random' else DQN.load(self.model_dir / agent, env=env, device='cpu')
            meanings = env.env_method('get_action_meanings')[0]
            episode, last_reward = 0, None
            while not self.stop_event.is_set():
                env.seed(seed + episode)
                env.action_space.seed(seed + episode)
                observation = env.reset()
                episode += 1
                total, length = 0.0, 0
                self._publish(env.get_images()[0], status='playing', episode=episode,
                              reward=0, length=0, action='—', last_reward=last_reward,
                              lives=env.venv.venv.envs[0].unwrapped.ale.lives())
                while not self.stop_event.is_set():
                    if self.pause_event.is_set():
                        self.stop_event.wait(0.05)
                        continue
                    started = time.monotonic()
                    action = [env.action_space.sample()] if model is None else model.predict(observation, deterministic=True)[0]
                    observation, rewards, dones, infos = env.step(action)
                    total += float(rewards[0])
                    length += 1
                    label = meanings[int(action[0])]
                    if infos[0].get('ale_auto_fire'):
                        label += ' + 自动发球'
                    values = dict(reward=total, length=length, action=label,
                                  lives=infos[0].get('lives'), status='playing')
                    if dones[0]:
                        # The VecEnv already reset; keep the last game's final visible frame.
                        last_reward = total
                        with self.lock:
                            self.state.update(values, status='episode_end', last_reward=last_reward)
                        self.stop_event.wait(1.5)
                        break
                    self._publish(env.get_images()[0], **values)
                    # 60 ALE frames / skip 4 = 15 policy decisions per second.
                    self.stop_event.wait(max(0, 1 / 15 - (time.monotonic() - started)))
        except Exception as error:
            with self.lock:
                self.state.update(status='error', error=str(error))
        finally:
            if env is not None:
                env.close()
