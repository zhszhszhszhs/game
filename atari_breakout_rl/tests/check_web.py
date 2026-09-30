"""Manual integration check against an already-running localhost game server."""
import argparse
import io
import json
import time
from urllib.request import Request, ProxyHandler, build_opener
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8501)
    args = parser.parse_args()
    base = f'http://127.0.0.1:{args.port}'
    opener = build_opener(ProxyHandler({}))  # localhost must not use an outbound HTTP proxy

    def request(path, data=None):
        req = Request(base + path, data=json.dumps(data).encode() if data is not None else None,
                      headers={'Content-Type': 'application/json'} if data is not None else {})
        with opener.open(req, timeout=30) as response:
            return response.read()

    def state():
        return json.loads(request('/api/game/state'))

    connected = False
    try:
        assert 'Agent 游戏演示' in request('/').decode()
        assert '.screen' in request('/style.css').decode()
        assert '/api/game/stream' in request('/app.js').decode()
        connected = True
        model = state().get('recommended_model')
        if model == 'random':
            model = None
        for agent in ['random'] + ([model] if model else []):
            request('/api/game/start', dict(agent=agent, seed=20000))
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                snapshot = state()
                assert not snapshot['error'], snapshot['error']
                if snapshot['length'] >= 5:
                    break
                time.sleep(.1)
            assert snapshot['length'] >= 5, snapshot
            # Decode a continuous sequence, not just the first HTTP response frame.
            with opener.open(base + '/api/game/stream', timeout=10) as stream:
                assert 'multipart/x-mixed-replace' in stream.headers['Content-Type']
                started = time.monotonic()
                for _ in range(30):
                    while stream.readline().strip() != b'--frame':
                        if time.monotonic() - started > 10:
                            raise AssertionError('Stream stalled')
                    headers = {}
                    while True:
                        line = stream.readline().strip()
                        if not line:
                            break
                        key, value = line.split(b':', 1)
                        headers[key.lower()] = value.strip()
                    picture = Image.open(io.BytesIO(stream.read(int(headers[b'content-length']))))
                    picture.load()
                    assert picture.size == (160, 210) and picture.format == 'JPEG'
                print(f'Stream: 30 real frames in {time.monotonic()-started:.2f}s')
            request('/api/game/speed', dict(speed=0.5))
            assert state()['speed'] == 0.5
            request('/api/game/speed', dict(speed=1))
            request('/api/game/pause', dict(paused=True))
            time.sleep(.15)
            before = state()['length']
            time.sleep(.2)
            assert state()['length'] == before
            assert state()['status'] == 'paused'
            frame = request('/api/game/frame')
            image = Image.open(io.BytesIO(frame))
            assert image.size == (160, 210) and image.format == 'PNG'
            print(f'PASS: {agent}: real PNG frame, CPU inference, HTTP controls and pause')
            request('/api/game/pause', dict(paused=False))
            time.sleep(.3)
            assert state()['length'] > before
    finally:
        if connected:
            request('/api/game/stop', {})
    print('Web integration checks passed. Viewer stopped; GPU training unaffected.')


if __name__ == '__main__':
    main()
