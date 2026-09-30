"""Start training in a separate session, or inspect its actual PID and runtime state."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--log-dir', type=Path, default=ROOT / 'logs')
    parser.add_argument('--status', action='store_true')
    args, remaining = parser.parse_known_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name):
        parser.error('run-name must be a simple directory name')
    logs = args.log_dir.resolve()
    manifest = logs / f'{args.run_name}.launch.json'
    if args.status:
        data = json.loads(manifest.read_text())
        try:
            os.kill(data['pid'], 0)
            command = Path(f"/proc/{data['pid']}/cmdline").read_bytes()
            data['process_running'] = str(ROOT / 'src/train.py').encode() in command
        except (ProcessLookupError, FileNotFoundError):
            data['process_running'] = False
        runtime = logs / args.run_name / 'runtime.json'
        if runtime.exists():
            data['runtime'] = json.loads(runtime.read_text())
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return
    if manifest.exists() or (logs / args.run_name).exists():
        parser.error('Run already exists; use a new run-name or --status')
    logs.mkdir(parents=True, exist_ok=True)
    console = logs / f'{args.run_name}.console.log'
    command = [sys.executable, '-u', str(ROOT / 'src/train.py'),
               '--run-name', args.run_name, '--log-dir', str(logs), *remaining]
    with console.open('x') as output:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    data = dict(pid=process.pid, command=command, console=str(console),
                started_at=datetime.now(timezone.utc).isoformat())
    manifest.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
