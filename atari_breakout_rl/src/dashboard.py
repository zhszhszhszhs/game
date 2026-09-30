"""Live ALE game interface, served on localhost for SSH port forwarding."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from game_viewer import GameViewer

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent / 'web'


def make_handler(viewer):
    class Handler(BaseHTTPRequestHandler):
        def send(self, content, kind='application/json; charset=utf-8', status=200):
            body = content if isinstance(content, bytes) else json.dumps(content, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/':
                self.send((WEB / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            elif path == '/api/game/state':
                self.send(viewer.snapshot())
            elif path == '/api/game/frame':
                frame = viewer.image()
                self.send(frame or b'', 'image/png', status=200 if frame else 204)
            else:
                self.send(dict(error='Not found'), status=404)

        def do_POST(self):
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                self.send(dict(error='JSON required'), status=415)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size < 4096:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('JSON object required')
                path = urlparse(self.path).path
                if path == '/api/game/start':
                    viewer.start(body.get('agent', 'random'), body.get('seed', 20000))
                elif path == '/api/game/pause':
                    viewer.pause(bool(body.get('paused', True)))
                elif path == '/api/game/stop':
                    viewer.stop()
                else:
                    self.send(dict(error='Not found'), status=404)
                    return
                self.send(viewer.snapshot())
            except (ValueError, RuntimeError, TypeError) as error:
                self.send(dict(error=str(error)), status=400)

        def log_message(self, format, *args):
            if len(args) > 1 and str(args[1]) not in ('200', '204', '304'):
                super().log_message(format, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=['127.0.0.1', 'localhost'], default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8501)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models')
    args = parser.parse_args()
    viewer = GameViewer(args.model_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(viewer))
    print(f'Game viewer: http://{args.host}:{args.port} (SSH forwarding supported)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.stop()
        server.server_close()


if __name__ == '__main__':
    main()
