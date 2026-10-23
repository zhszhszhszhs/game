"""Independent, constant-speed Breakout. No ALE/ROM code is executed here.

Single-player rules follow the Atari 2600 manual (5 balls, 6x18 bricks,
7/7/4/4/1/1 points, two walls). Geometry and bounce angles are a documented
reconstruction, not a cycle-accurate ROM reproduction. Time is fixed at 60 Hz.
"""
import math

import gymnasium as gym
import numpy as np


class FixedSpeedBreakout(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 60}
    WIDTH, HEIGHT = 160, 210
    LEFT, RIGHT, TOP = 8.0, 152.0, 32.0
    PADDLE_Y, PADDLE_WIDTH, PADDLE_SPEED = 189.0, 16.0, 3.0
    BALL_SPEED, RADIUS = 2.0, 1.0  # pixels per physics frame; 120 pixels/second
    ROW_POINTS = np.array([7, 7, 4, 4, 1, 1])
    COLORS = [(200, 72, 72), (198, 108, 58), (180, 156, 48),
              (72, 160, 72), (66, 158, 130), (66, 114, 194)]

    def __init__(self, render_mode=None, max_frames=108000, physics_version=3):
        super().__init__()
        if render_mode not in (None, 'rgb_array'):
            raise ValueError('Use rgb_array and the web viewer for this environment')
        if max_frames < 1:
            raise ValueError('max_frames must be positive')
        if physics_version != 3:
            raise ValueError('Unsupported physics version')
        self.physics_version = physics_version
        self.render_mode, self.max_frames = render_mode, max_frames
        self.observation_space = gym.spaces.Box(0, 255, (210, 160, 3), np.uint8)
        self.action_space = gym.spaces.Discrete(4)
        self._rects = np.array([(8 + c*8, 57 + r*5, 15 + c*8, 61 + r*5)
                                for r in range(6) for c in range(18)], dtype=float)
        self._background = np.zeros((210, 160, 3), dtype=np.uint8)
        self._background[32:196, 4:8] = 142
        self._background[32:196, 152:156] = 142
        self._background[28:32, 4:156] = 142
        self._initialized = False

    def get_action_meanings(self):
        return ['NOOP', 'FIRE', 'RIGHT', 'LEFT']

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.lives, self.score, self.wall, self.frames = 5, 0, 1, 0
        self.bricks = np.ones(108, dtype=bool)
        self.paddle_x = 80.0
        self._done = False
        self._initialized = True
        self._prepare_serve()
        return self.render(), self._info()

    def _prepare_serve(self):
        self.paddle_width = self.PADDLE_WIDTH
        self.paddle_hits = 0
        self.waiting_for_fire = True
        self.ball = np.array([80.0, 120.0])
        self.velocity = np.zeros(2, dtype=float)

    def _serve(self):
        # Seeded launch position/direction vary games without changing speed.
        self.ball[:] = [self.np_random.uniform(50, 110), 120]
        self._direction(float(self.np_random.choice([-1, 1])), 2.0)
        self.waiting_for_fire = False

    def _direction(self, x, y):
        self.velocity[:] = np.array([x, y]) * (self.BALL_SPEED / math.hypot(x, y))

    def _info(self, **extra):
        return dict(lives=self.lives, score=self.score, wall=self.wall,
                    ball_speed=float(np.linalg.norm(self.velocity)),
                    configured_ball_speed=self.BALL_SPEED,
                    waiting_for_fire=self.waiting_for_fire, paddle_hits=self.paddle_hits,
                    paddle_width=self.paddle_width, physics_version=self.physics_version,
                    frames=self.frames, **extra)

    def _brick_collision(self, remaining):
        """Swept square against expanded brick rectangles: no tunnelling."""
        indices = np.flatnonzero(self.bricks)
        if not len(indices):
            return None
        rects = self._rects[indices]
        lo, hi = rects[:, :2] - self.RADIUS, rects[:, 2:] + self.RADIUS
        enter = np.full((len(indices), 2), -np.inf)
        leave = np.full_like(enter, np.inf)
        for axis in (0, 1):
            if abs(self.velocity[axis]) < 1e-12:
                outside = (self.ball[axis] < lo[:, axis]) | (self.ball[axis] > hi[:, axis])
                enter[outside, axis] = np.inf
            else:
                a = (lo[:, axis] - self.ball[axis]) / self.velocity[axis]
                b = (hi[:, axis] - self.ball[axis]) / self.velocity[axis]
                enter[:, axis], leave[:, axis] = np.minimum(a, b), np.maximum(a, b)
        times = enter.max(axis=1)
        valid = (times >= -1e-9) & (times <= remaining + 1e-9) & (times <= leave.min(axis=1))
        if not valid.any():
            return None
        earliest = float(times[valid].min())
        tied = valid & np.isclose(times, earliest, atol=1e-9, rtol=0)
        axes = np.any(np.isclose(enter[tied], earliest, atol=1e-9, rtol=0), axis=0)
        return max(0.0, earliest), indices[tied], axes

    def _bounce_paddle(self):
        self.paddle_hits += 1
        offset = (self.ball[0] - self.paddle_x) / (self.paddle_width / 2)
        section = int(np.clip((offset + 1) * 2.5, 0, 4))
        angle = [-60.0, -30.0, 0.0, 30.0, 60.0][section]
        # Hit count is telemetry only; the rebound rule is constant.
        if section == 2:
            angle = 10.0 if self.velocity[0] >= 0 else -10.0
        radians = math.radians(angle)
        self._direction(math.sin(radians), -math.cos(radians))

    def _advance_ball(self):
        remaining, reward, lost = 1.0, 0, False
        # At 2 px/frame, this bound exceeds physically reachable contacts.
        for _ in range(16):
            if remaining < 1e-10:
                break
            candidates = []
            vx, vy = self.velocity
            x, y = self.ball
            if vx < 0:
                candidates.append(((self.LEFT + self.RADIUS - x) / vx, 'x'))
            elif vx > 0:
                candidates.append(((self.RIGHT - self.RADIUS - x) / vx, 'x'))
            if vy < 0:
                candidates.append(((self.TOP + self.RADIUS - y) / vy, 'ceiling'))
            elif vy > 0:
                t = (self.PADDLE_Y - self.RADIUS - y) / vy
                if t >= -1e-9 and abs(x + vx*t - self.paddle_x) <= self.paddle_width/2 + self.RADIUS:
                    candidates.append((t, 'paddle'))
                candidates.append(((self.HEIGHT + self.RADIUS - y) / vy, 'lost'))
            brick = self._brick_collision(remaining)
            if brick is not None:
                candidates.append((brick[0], 'brick'))
            candidates = [(max(0.0, t), kind) for t, kind in candidates
                          if -1e-9 <= t <= remaining + 1e-9]
            if not candidates:
                self.ball += self.velocity * remaining
                break
            t = min(t for t, _ in candidates)
            kinds = {kind for time, kind in candidates if abs(time-t) < 1e-9}
            self.ball += self.velocity * t
            remaining -= t
            if 'lost' in kinds:
                self.lives -= 1
                lost = True
                if self.lives:
                    self._prepare_serve()
                else:
                    self._done = True
                    self.velocity[:] = 0
                break
            flip = np.array(['x' in kinds, 'ceiling' in kinds])
            if 'brick' in kinds:
                _, indices, axes = brick
                self.bricks[indices] = False
                reward += int(self.ROW_POINTS[indices // 18].sum())
                flip |= axes
            self.velocity[flip] *= -1
            if 'paddle' in kinds:
                self._bounce_paddle()
                if 'x' in kinds:
                    self.velocity[0] = abs(self.velocity[0]) if x < 80 else -abs(self.velocity[0])
            # Separate contact surfaces at a negligible subpixel distance.
            self.ball += self.velocity * 1e-8
            if not self.bricks.any():
                if self.wall == 2:
                    self._done = True
                else:
                    self.wall = 2
                    self.bricks[:] = True
                    # Serve the second wall to avoid respawning bricks around
                    # the ball. This is an explicit reconstruction difference.
                    self._prepare_serve()
                break
        else:
            raise RuntimeError('Collision iteration limit exceeded')
        self.score += reward
        return reward, lost

    def step(self, action):
        if not self._initialized or self._done:
            raise RuntimeError('Call reset() before stepping a new/finished game')
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action: {action}')
        self.frames += 1
        dx = self.PADDLE_SPEED * (int(action == 2) - int(action == 3))
        self.paddle_x = float(np.clip(self.paddle_x + dx,
                                     self.LEFT + self.paddle_width/2, self.RIGHT - self.paddle_width/2))
        if self.waiting_for_fire and action == 1:
            self._serve()
        reward, lost = (0, False) if self.waiting_for_fire else self._advance_ball()
        terminated = self._done
        truncated = self.frames >= self.max_frames and not terminated
        self._done = terminated or truncated
        return self.render(), float(reward), terminated, truncated, self._info(life_lost=lost)

    def render(self):
        if not self._initialized:
            raise RuntimeError('Call reset() before render()')
        frame = self._background.copy()
        for index in np.flatnonzero(self.bricks):
            x0, y0, x1, y1 = self._rects[index].astype(int)
            frame[y0:y1, x0:x1] = self.COLORS[index // 18]
        left = int(round(self.paddle_x - self.paddle_width/2))
        frame[189:193, left:left+int(self.paddle_width)] = (210, 210, 210)
        if not self.waiting_for_fire and self.lives:
            x, y = np.rint(self.ball).astype(int)
            frame[max(0, y-1):min(210, y+2), max(0, x-1):min(160, x+2)] = 240
        # Score and lives are also visible in the observation, as in Atari.
        import cv2
        cv2.putText(frame, f'{self.score:03d}', (9, 19), cv2.FONT_HERSHEY_PLAIN, 1, (210, 210, 210), 1)
        cv2.putText(frame, str(self.lives), (139, 19), cv2.FONT_HERSHEY_PLAIN, 1, (210, 210, 210), 1)
        return frame


class AutoFire(gym.Wrapper):
    """Fire on the next raw frame when waiting; never add hidden extra frames."""
    def step(self, action):
        fire = self.unwrapped.waiting_for_fire
        obs, reward, terminated, truncated, info = self.env.step(1 if fire else action)
        info['auto_fire'] = fire
        return obs, reward, terminated, truncated, info


class RepeatAndPool(gym.Wrapper):
    """Action repeat/max pool with correct terminal/truncation observations.

    SB3's Atari MaxAndSkipEnv may return old frames on an early terminal step.
    Here the actual last two frames are pooled, including at time limits.
    """
    def step(self, action):
        frames, total, auto_fire, lost = [], 0.0, False, False
        paddle_contacts, lives_lost = 0, 0
        for _ in range(4):
            previous_hits, previous_lives = self.unwrapped.paddle_hits, self.unwrapped.lives
            obs, reward, terminated, truncated, info = self.env.step(action)
            paddle_contacts += max(0, info['paddle_hits'] - previous_hits)
            lives_lost += max(0, previous_lives - info['lives'])
            frames = (frames + [obs])[-2:]
            total += reward
            auto_fire |= info.get('auto_fire', False)
            lost |= info.get('life_lost', False)
            if terminated or truncated:
                break
        info = dict(info, auto_fire=auto_fire, life_lost=lost,
                    paddle_contacts=paddle_contacts, lives_lost=lives_lost)
        return np.maximum(frames[0], frames[-1]), total, terminated, truncated, info
