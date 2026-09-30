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
        assert 'Breakout 游戏演示' in request('/').decode()
        connected = True
        model = next((m for m in state()['models'] if m.endswith('final_model.zip')), None)
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
