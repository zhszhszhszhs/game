"""Build PPO/report comparison evidence from real scores and immutable models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

from generate_figures import configure, save, TITLE_FONT, PANEL_FONT

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'report/data'
sys.path.insert(0, str(ROOT / 'src'))
from env import environment_spec

RUNS = {
    'ppo98': 'ppo_custom_fixed_angle_gae_control_500k_20261022',
    'ppo99': 'ppo_custom_fixed_angle_gae_500k_20261022',
}
LABELS = {'random': '随机策略', 'dqn': 'DQN', 'source': 'PPO 共同起点',
          'ppo98': 'PPO λ=0.98', 'ppo99': 'PPO λ=0.99'}
COLORS = {'dqn': '#3366a0', 'ppo98': '#267f70', 'ppo99': '#b76d33',
          'source': '#8793a2', 'random': '#a6adb7'}


def architecture():
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.set(xlim=(0, 10), ylim=(0, 4))
    ax.axis('off')
    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.08',
            facecolor=color, edgecolor='#94a3b8', linewidth=1))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=11)
    def arrow(a, b):
        ax.annotate('', xy=b, xytext=a,
                    arrowprops=dict(arrowstyle='->', color='#475569', lw=1.6))
    box(.1, 1.3, 1.6, 1, '四帧图像\n4×84×84\n像素 /255', '#edf2f7')
    box(2.4, 2.35, 2.4, .9, '策略 CNN\nNatureCNN → 512', '#dceaf7')
    box(2.4, .3, 2.4, .9, '价值 CNN\nNatureCNN → 512', '#dcefe9')
    box(5.55, 2.35, 1.8, .9, '4 个 logits\nSoftmax 分布', '#dceaf7')
    box(5.55, .3, 1.8, .9, '1 个标量\nV(s)', '#dcefe9')
    box(8, 2.35, 1.75, .9, '训练：采样动作\n评估：最大概率', '#edf2f7')
    box(8, .3, 1.75, .9, 'GAE / returns\n价值回归目标', '#edf2f7')
    arrow((1.7, 1.8), (2.35, 2.8)); arrow((1.7, 1.8), (2.35, .75))
    arrow((4.8, 2.8), (5.5, 2.8)); arrow((4.8, .75), (5.5, .75))
    arrow((7.35, 2.8), (7.95, 2.8)); arrow((7.35, .75), (7.95, .75))
    ax.text(5.1, 1.7, '策略梯度裁剪 0.5       价值梯度裁剪 1.0',
            ha='center', va='center', fontsize=11)
    fig.suptitle('PPO 策略与价值网络：独立特征、共同图像输入', fontproperties=TITLE_FONT)
    fig.tight_layout(rect=(0, 0, 1, .92))
    save(fig, 'ppo_actor_critic.png')


def training():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    evidence = {}
    for key, run in RUNS.items():
        base = ROOT / 'logs' / run
        with np.load(base / 'eval/evaluations.npz') as values:
            steps = values['timesteps'] / 1000
            rewards = values['results']
            axes[0, 0].plot(steps, rewards.mean(axis=1), 'o-', color=COLORS[key], label=LABELS[key])
            axes[0, 1].plot(steps, np.median(rewards, axis=1), 'o-', color=COLORS[key], label=LABELS[key])
            evidence[key] = dict(run=run, steps=values['timesteps'].tolist(),
                mean=rewards.mean(axis=1).tolist(), median=np.median(rewards,axis=1).tolist())
        log = pd.read_csv(base / 'progress.csv')
        for ax, column in [(axes[1, 0], 'rollout/ep_rew_mean'),
                           (axes[1, 1], 'train/policy_kl_after_update')]:
            x = log[['time/total_timesteps', column]].dropna()
            ax.plot(x['time/total_timesteps']/1000, x[column], color=COLORS[key], alpha=.85, label=LABELS[key])
    for ax, title, ylabel in zip(axes.flat,
            ['固定 32 局开发均分', '固定 32 局开发中位数', '训练时近 100 局采样均分', '整轮更新后的近似 KL'],
            ['原始得分', '原始得分', '原始得分', '近似 KL']):
        ax.set_title(title, fontproperties=PANEL_FONT)
        ax.set(xlabel='本阶段新增决策步 / 千步', ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=9)
    axes[1, 1].set_yscale('log')
    fig.suptitle('PPO 单因素对照：相同起点、预算与开发种子', fontproperties=TITLE_FONT)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, 'ppo_training_curves.png')
    (DATA/'ppo_training_sources.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')


def evidence():
    root = ROOT / 'results/evaluation'
    sources = {
        'random': root/'report_dqn_ppo_50/dqn/random_results.csv',
        'dqn': root/'report_dqn_ppo_50/dqn/dqn_results.csv',
        'source': root/'ppo_gae_20261022/source_heldout/ppo_results.csv',
        **{key: root/'ppo_gae_20261022'/run/'ppo_results.csv' for key,run in RUNS.items()},
    }
    seeds = np.arange(121000,121050)
    frames = {}
    metrics = {}
    manifest = {'episodes':50,'first_seed':121000,'last_seed':121049,
        'bootstrap_replicates':20000,'bootstrap_seed':20261022,
        'primary_ppo':'ppo98',
        'selection_reason':'Primary PPO chosen by the larger development mean (196 versus 171.09375), not by test scores.',
        'environment':environment_spec(),'sources':{},'metrics':{},'paired':{},
        'scope':'Fixed trained policies; unequal multi-stage training histories. Episode bootstrap does not measure training-seed variability.'}
    for key,path in sources.items():
        frame = pd.read_csv(path).sort_values('seed').reset_index(drop=True)
        np.testing.assert_array_equal(frame.seed,seeds)
        if len(frame)!=50 or frame.seed.duplicated().any() or not np.isfinite(frame.reward).all():
            raise ValueError(f'Invalid evaluation: {path}')
        values=frame.reward.to_numpy(float)
        frames[key]=frame
        metrics[key]={'label':LABELS[key],'mean':float(values.mean()),'std':float(values.std()),
            'median':float(np.median(values)),'p25':float(np.percentile(values,25)),
            'below50':int(np.sum(values<50)),'below50_rate':float(np.mean(values<50)),
            'min':float(values.min()),'max':float(values.max()),
            'truncated':int(frame.truncated.astype(str).str.lower().eq('true').sum()),
            'mean_length':float(frame.length.mean()),'completed_walls':int(np.sum(values==864))}
        record={'csv':str(path.relative_to(ROOT)),'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        meta_path=path.with_suffix('.json')
        if key!='random':
            meta=json.loads(meta_path.read_text())
            if meta['environment']!=environment_spec():raise ValueError('Environment mismatch')
            model=Path(meta['model'])
            record.update(model=str(model.relative_to(ROOT)),model_sha256=hashlib.sha256(model.read_bytes()).hexdigest())
        manifest['sources'][key]=record
    paired=pd.DataFrame({'seed':seeds,**{key:frame.reward for key,frame in frames.items()}})
    rng=np.random.default_rng(20261022)
    for a,b in [('dqn','ppo98'),('dqn','ppo99'),('ppo99','ppo98'),('ppo99','source'),('ppo98','source')]:
        delta=(paired[a]-paired[b]).to_numpy(float)
        boot=delta[rng.integers(0,50,(20000,50))].mean(axis=1)
        manifest['paired'][a+'_minus_'+b]={'mean':float(delta.mean()),
            'ci95':np.percentile(boot,[2.5,97.5]).tolist(),
            'wins':int(np.sum(delta>0)),'ties':int(np.sum(delta==0)),'losses':int(np.sum(delta<0))}
    manifest['metrics']=metrics
    paired.to_csv(DATA/'dqn_ppo_paired_scores.csv',index=False)
    pd.DataFrame(metrics).T.to_csv(DATA/'dqn_ppo_metrics.csv',index_label='model')
    (DATA/'ppo_report_summary.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest,paired


def comparison(manifest, paired):
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),gridspec_kw={'width_ratios':[1.15,1]})
    keys=['dqn','ppo98','ppo99']
    axes[0].boxplot([paired[k] for k in keys],tick_labels=[LABELS[k] for k in keys],
                    showmeans=True,showfliers=False,meanprops=dict(marker='D',markerfacecolor='#be5046',markersize=5))
    rng=np.random.default_rng(47)
    for i,key in enumerate(keys,1):
        axes[0].scatter(i+rng.uniform(-.13,.13,50),paired[key],s=13,alpha=.45,color=COLORS[key])
    axes[0].set(ylabel='原始游戏得分',title='相同 50 局的分布、均值与中位数')
    delta=paired.dqn-paired.ppo98
    axes[1].bar(np.arange(1,51),np.sort(delta),width=.85,color=np.where(np.sort(delta)>=0,COLORS['dqn'],'#b76d33'))
    axes[1].axhline(0,color='#64748b',lw=1)
    axes[1].axhline(delta.mean(),color='#b24545',ls='--',label=f'平均差 {delta.mean():.2f}')
    axes[1].set(xlabel='按配对差排序的种子序号',ylabel='DQN − PPO λ=0.98 得分',title='固定模型的逐局配对差')
    axes[1].legend(frameon=False,fontsize=9)
    for ax in axes:
        ax.title.set_fontproperties(PANEL_FONT)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('DQN 与 PPO：统一任务和 50 个测试种子',fontproperties=TITLE_FONT)
    fig.tight_layout(rect=(0,0,1,.94))
    save(fig,'dqn_ppo_comparison.png')


def tex_summary(manifest):
    lines=['% Generated by report/generate_ppo_figures.py from measured scores.']
    names={'random':'Random','dqn':'DQN','source':'Source','ppo98':'PPOControl','ppo99':'PPOGae'}
    for key,name in names.items():
        m=manifest['metrics'][key]
        for field,suffix in [('mean','Mean'),('std','Std'),('median','Median'),('p25','Pxxv'),('max','Max')]:
            lines.append('\\newcommand{\\'+name+suffix+'}{'+f'{m[field]:.2f}'+'}')
        for field,suffix in [('below50','Low'),('truncated','Timeout')]:
            lines.append('\\newcommand{\\'+name+suffix+'}{'+str(m[field])+'}')
    for key,name in [('dqn_minus_ppo98','DQNPPO'),('ppo99_minus_ppo98','GaeControl')]:
        stats=manifest['paired'][key]
        lines.append('\\newcommand{\\'+name+'Diff}{'+f"{stats['mean']:.2f}"+'}')
        lines.append('\\newcommand{\\'+name+'CI}{'+f"[{stats['ci95'][0]:.2f},{stats['ci95'][1]:.2f}]"+'}')
        for field in ['wins','ties','losses']:
            lines.append('\\newcommand{\\'+name+field.title()+'}{'+str(stats[field])+'}')
    (ROOT/'report/latex/comparison_metrics.tex').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',choices=['all','architecture','training','comparison'],default='all')
    args=parser.parse_args()
    configure()
    if args.only in ['all','architecture']:architecture()
    if args.only in ['all','training']:training()
    if args.only in ['all','comparison']:
        manifest,paired=evidence()
        comparison(manifest,paired)
        tex_summary(manifest)
        print(json.dumps(manifest['metrics'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
