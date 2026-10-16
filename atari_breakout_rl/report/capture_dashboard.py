"""Capture the actual local dashboard with Firefox's built-in Marionette API.

Start src/dashboard.py --port 8502 first. No browser packages are required.
"""
import base64
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:8502'


def api(path, data=None):
    req = urllib.request.Request(BASE + path)
    if data is not None:
        req.data = json.dumps(data).encode()
        req.add_header('Content-Type', 'application/json')
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.load(response)


class Marionette:
    def __init__(self, sock):
        self.sock, self.counter = sock, 0
        self.read()

    def read(self):
        size = b''
        while not size.endswith(b':'):
            byte = self.sock.recv(1)
            if not byte:
                raise RuntimeError('Browser disconnected')
            size += byte
        remaining = int(size[:-1])
        chunks = []
        while remaining:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise RuntimeError('Incomplete browser response')
            chunks.append(chunk)
            remaining -= len(chunk)
        return json.loads(b''.join(chunks))

    def command(self, name, params=None):
        self.counter += 1
        body = json.dumps([0, self.counter, name, params or {}]).encode()
        self.sock.sendall(str(len(body)).encode() + b':' + body)
        response = self.read()
        if response[2]:
            raise RuntimeError(response[2])
        return response[3]

    def js(self, script):
        return self.command('WebDriver:ExecuteScript', {'script': script, 'args': [], 'newSandbox': False})['value']


def main():
    api('/api/game/start', {'agent': 'dqn_custom_fixed_angle_refine_200k/best_model/best_model.zip', 'seed': 42000})
    api('/api/game/speed', {'speed': 2})
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        state = api('/api/game/state')
        if state['status'] == 'error':
            raise RuntimeError(state['error'])
        if state['reward'] >= 20:
            break
        time.sleep(.25)
    else:
        raise RuntimeError('Model did not produce a scored game frame')
    api('/api/game/pause', {'paused': True})
    time.sleep(.2)
    state = api('/api/game/state')
    (ROOT / 'report/data').mkdir(exist_ok=True)
    (ROOT / 'report/data/dashboard_state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='breakout-report-firefox-') as tmp:
        profile = Path(tmp)
        (profile / 'user.js').write_text('user_pref("marionette.port", 2829);\n'
                                       'user_pref("browser.shell.checkDefaultBrowser", false);\n'
                                       'user_pref("browser.startup.homepage_override.mstone", "ignore");\n')
        with (ROOT / 'report/data/firefox.log').open('w') as log:
            process = subprocess.Popen(['firefox', '--headless', '--no-remote', '--marionette',
                                        '--profile', str(profile), 'about:blank'], stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 30
                while True:
                    try:
                        sock = socket.create_connection(('127.0.0.1', 2829), timeout=5)
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise
                        time.sleep(.25)
                sock.settimeout(30)
                browser = Marionette(sock)
                browser.command('WebDriver:NewSession', {'capabilities': {'alwaysMatch': {}}})
                browser.command('WebDriver:SetWindowRect', {'width': 1360, 'height': 1200})
                browser.command('WebDriver:Navigate', {'url': BASE})
                deadline = time.monotonic() + 30
                while not browser.js('return document.getElementById("game").naturalWidth > 0 && '
                                     'document.getElementById("reward").textContent !== "0";'):
                    if time.monotonic() > deadline:
                        raise RuntimeError('Dashboard frame did not render')
                    time.sleep(.25)
                # Hide footer notes only in this report capture; keep app source intact.
                browser.js('const style = document.createElement("style");'
                           'style.textContent = "footer,.details,.arena-footer,.history-panel{display:none!important}'
                           'h1{font-family:\\"Noto Sans CJK SC\\",sans-serif;font-size:34px;font-weight:700}";'
                           'document.head.appendChild(style); return true;')
                height = browser.js('return Math.ceil(document.querySelector("main").getBoundingClientRect().bottom'
                                    ' + window.outerHeight - window.innerHeight);')
                browser.command('WebDriver:SetWindowRect', {'width': 1360, 'height': height})
                result = browser.command('WebDriver:TakeScreenshot', {'full': True})
                output = ROOT / 'report/latex/images/game_live.png'
                output.write_bytes(base64.b64decode(result['value']))
                print(f'Saved {output}; score={state["reward"]}, lives={state["lives"]}', flush=True)
                browser.command('Marionette:Quit', {'flags': ['eForceQuit']})
                sock.close()
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    api('/api/game/stop', {})


if __name__ == '__main__':
    main()
