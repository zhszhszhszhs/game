"""Freeze and queue a paired, three-seed adaptive-KL PPO continuation study."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from env import environment_spec
from utils import ROOT, write_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(name, device, workers, smoke=False):
    output = ROOT/'results/evaluation'/name
    output.mkdir(parents=True, exist_ok=False)
    frozen = output/'frozen'
    (frozen/'src').mkdir(parents=True)
    for source in (ROOT/'src').glob('*.py'):
        shutil.copy2(source, frozen/'src'/source.name)
    config = frozen/'config.yaml'
    shutil.copy2(ROOT/'configs/ppo_custom_adaptive_kl.yaml', config)
    origin = ROOT/'models/ppo_custom_fixed_angle_gae_control_500k_20261022/best_model/best_model.zip'
    source_dir = ROOT/'models'/f'{name}_source'
    source_dir.mkdir(parents=True, exist_ok=False)
    source_model = source_dir/'source.zip'
    shutil.copy2(origin, source_model)
    seeds = [48] if smoke else [48,49,50]
    eval_seed, test_seed = (221000,231000) if smoke else (131000,141000)
    jobs = []
    for seed in seeds:
        for arm in ('adaptive','control'):
            run = f'{name}_{arm}_s{seed}'
            command = [sys.executable,'-u',str(frozen/'src/train_ppo.py'),
                '--config',str(config),'--resume',str(source_model),'--run-name',run,
                '--log-dir',str(ROOT/'logs'),'--model-dir',str(ROOT/'models'),
                '--evaluation-dir',str(output),'--seed',str(seed),'--device',device,
                '--eval-seed',str(eval_seed),'--post-eval-seed',str(test_seed),
                '--adaptive-kl-enabled' if arm=='adaptive' else '--no-adaptive-kl-enabled']
            if smoke:
                command += ['--total-timesteps','32','--n-envs','2','--n-steps','8',
                    '--batch-size','8','--n-epochs','1','--eval-envs','2','--n-eval-episodes','2',
                    '--post-eval-episodes','2','--eval-freq','32','--checkpoint-freq','32',
                    '--threads','1','--no-evaluate-at-start']
            jobs.append(dict(name=run,kind='training',arm=arm,seed=seed,command=command,
                results=str(output/run/'ppo_results.csv')))
    jobs.append(dict(name=f'{name}_source_test',kind='source_evaluation',command=[
        sys.executable,'-u',str(frozen/'src/evaluate.py'),'--model',str(source_model),
        '--episodes','2' if smoke else '50','--seed',str(test_seed),'--device','cpu',
        '--threads','1','--output-dir',str(output/'source_heldout')],
        results=str(output/'source_heldout/ppo_results.csv')))
    protocol = dict(name=name,created_at=datetime.now(timezone.utc).isoformat(),
        project_root=str(ROOT),environment=environment_spec(),device=device,workers=workers,
        smoke=smoke,training_seeds=seeds,development_seed=eval_seed,
        development_episodes=2 if smoke else 32,test_seed=test_seed,test_episodes=2 if smoke else 50,
        planned_steps_per_run=32 if smoke else 500000,
        actual_rollout_rounded_steps=32 if smoke else 503808,
        source_origin=str(origin),source=str(source_model),source_sha256=sha256(source_model),
        source_choice='Previous GAE lambda=.98 model chosen by development mean, no new test selection.',
        configuration=str(config),frozen_sha256={str(p):sha256(p) for p in frozen.rglob('*') if p.is_file()},
        single_factor='Actor-only bounded KL feedback enabled versus disabled; critic schedule unchanged.',
        selection='stable_near_mean; tolerance=.05; fixed development seeds; no early stopping.',
        kl_controller=dict(target=.0005,band=1.5,factor=1.1,ema_alpha=.2,
            min_actor_lr_multiplier=.5,max_actor_lr_multiplier=2.,
            measurement='Final sampled full-rollout KL; next update only; immediate downward spike response.'),
        scope='Three continuation seeds conditional on one frozen initialization; not from-scratch replication.',
        test_policy='No parameter changes or main-arm switching based on held-out scores.',
        bootstrap=dict(replicates=20000,seed=20261022,
            method='Paired two-way resampling of training-seed pairs and shared episode seeds.'),jobs=jobs)
    write_json(output/'protocol.json',protocol)
    print(json.dumps(dict(protocol=str(output/'protocol.json'),jobs=len(jobs),device=device),indent=2))


def paired_summary(adaptive, control, seed=20261022, replicates=20000):
    adaptive,control=np.asarray(adaptive,dtype=float),np.asarray(control,dtype=float)
    if adaptive.shape!=control.shape or adaptive.ndim!=2 or min(adaptive.shape)<1:
        raise ValueError('Expected matching training-seed by episode-seed matrices')
    if not (np.isfinite(adaptive).all() and np.isfinite(control).all()):
        raise ValueError('Non-finite evaluation score')
    delta=adaptive-control
    rng=np.random.default_rng(seed)
    train=rng.integers(delta.shape[0],size=(replicates,delta.shape[0]))
    games=rng.integers(delta.shape[1],size=(replicates,delta.shape[1]))
    samples=delta[train[:,:,None],games[:,None,:]].mean(axis=(1,2))
    return dict(mean_difference=float(delta.mean()),per_training_seed_mean_difference=delta.mean(axis=1).tolist(),
        ci95=np.percentile(samples,[2.5,97.5]).tolist(),
        training_seeds_won=int(np.sum(delta.mean(axis=1)>0)),
        caveat='Only three continuation seeds: interval is exploratory and conditional on frozen source.')


def summarize(protocol):
    seeds=np.arange(protocol['test_seed'],protocol['test_seed']+protocol['test_episodes'])
    matrices={arm:[] for arm in ('adaptive','control')}
    rows=[]
    for job in protocol['jobs']:
        frame=pd.read_csv(job['results']).sort_values('seed').reset_index(drop=True)
        np.testing.assert_array_equal(frame.seed,seeds)
        values=frame.reward.to_numpy(float)
        row=dict(run=job['name'],arm=job.get('arm','source'),training_seed=job.get('seed'),
            mean=float(values.mean()),median=float(np.median(values)),p25=float(np.percentile(values,25)),
            std=float(values.std()),below50=int(np.sum(values<50)),
            truncated=int(frame.truncated.astype(str).str.lower().eq('true').sum()),
            csv=job['results'],csv_sha256=sha256(job['results']))
        if job['kind']=='training':
            metadata=json.loads((ROOT/'logs'/job['name']/'selected_model.json').read_text())
            row.update(model=metadata['model'],model_sha256=sha256(metadata['model']))
            matrices[job['arm']].append(values)
        rows.append(row)
    comparison=paired_summary(matrices['adaptive'],matrices['control'],**{
        key:protocol['bootstrap'][key] for key in ('seed','replicates')})
    output=Path(protocol['configuration']).parents[1]
    pd.DataFrame(rows).to_csv(output/'comparison.csv',index=False)
    write_json(output/'comparison_summary.json',dict(status='completed',runs=rows,
        paired_adaptive_minus_control=comparison,scope=protocol['scope'],smoke=protocol['smoke']))


def run_study(protocol_path, name, log_dir):
    protocol=json.loads(protocol_path.read_text())
    if protocol['name']!=name or Path(protocol['project_root'])!=ROOT:
        raise ValueError('Study name/project mismatch')
    for path,digest in protocol['frozen_sha256'].items():
        if sha256(path)!=digest:raise ValueError(f'Frozen code/config changed: {path}')
    if sha256(protocol['source'])!=protocol['source_sha256']:
        raise ValueError('Frozen source model changed')
    directory=log_dir/name
    directory.mkdir(parents=True,exist_ok=False)
    lock=(directory/'study.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    queue=[dict(job,status='queued') for job in protocol['jobs']]
    running={}
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
        for process,handle in running.values():process.terminate()
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    environment=dict(os.environ,BREAKOUT_PROJECT_ROOT=str(ROOT))
    def heartbeat(status):
        records=[]
        for job in queue:
            record={k:v for k,v in job.items() if k!='command'}
            runtime=log_dir/job['name']/'runtime.json'
            if runtime.exists():
                record['runtime']=json.loads(runtime.read_text())
            records.append(record)
        write_json(directory/'runtime.json',dict(status=status,pid=os.getpid(),
            updated_at=datetime.now(timezone.utc).isoformat(),protocol=str(protocol_path),
            completed=sum(j['status']=='completed' for j in queue),total=len(queue),jobs=records))
    while True:
        failed=any(job['status']=='failed' for job in queue)
        for job in queue:
            if stopped or failed or len(running)>=protocol['workers']:break
            if job['status']!='queued':continue
            run_dir=log_dir/job['name']
            console=log_dir/f"{job['name']}.console.log"
            manifest=log_dir/f"{job['name']}.launch.json"
            if run_dir.exists() or console.exists() or manifest.exists():
                raise FileExistsError(f'Refusing to overwrite run: {job["name"]}')
            handle=console.open('x')
            process=subprocess.Popen(job['command'],cwd=ROOT,env=environment,
                stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
            running[job['name']]=(process,handle)
            job.update(status='running',pid=process.pid,console=str(console))
            write_json(manifest,dict(pid=process.pid,command=job['command'],console=str(console),
                script=job['command'][2],algorithm='ppo',study=name,
                started_at=datetime.now(timezone.utc).isoformat()))
            print(f'Started {job["name"]}: PID {process.pid}',flush=True)
        for job in queue:
            if job['name'] not in running:continue
            process,handle=running[job['name']]
            code=process.poll()
            if code is None:continue
            handle.close();del running[job['name']]
            job.update(status='completed' if code==0 else 'failed',exit_code=code)
            print(f'{job["status"]}: {job["name"]} (exit {code})',flush=True)
        if not running and (all(j['status']!='queued' for j in queue) or stopped or
                            any(j['status']=='failed' for j in queue)):break
        heartbeat('running')
        time.sleep(5)
    if stopped or any(j['status']!='completed' for j in queue):
        heartbeat('interrupted' if stopped else 'failed')
        raise RuntimeError('Study incomplete; inspect per-run console and runtime files')
    heartbeat('summarizing')
    try:
        summarize(protocol)
    except Exception:
        heartbeat('summary_failed')
        raise
    heartbeat('completed')
    print('Study completed; paired comparison_summary.json written',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--log-dir',type=Path,default=ROOT/'logs')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--protocol',type=Path)
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_name) or args.workers<1:
        parser.error('Require a simple run name and positive worker count')
    if args.prepare:prepare(args.run_name,args.device,args.workers,args.smoke)
    else:
        if args.protocol is None:parser.error('--protocol required to execute a frozen study')
        run_study(args.protocol.resolve(),args.run_name,args.log_dir.resolve())


if __name__=='__main__':main()
