"""Generate three PPO analysis figures from frozen, paired experiment evidence.

Default: validate original logs/results and export a reproducible data snapshot.
--from-snapshot: redraw only from report/data/ppo_analysis, without training logs.
No training, rollout sampling, or synthetic episode scores are performed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from generate_figures import configure, save, TITLE_FONT, PANEL_FONT

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'report/data/ppo_analysis'
STUDY = 'ppo_custom_fixed_angle_kl_study_20261022'
SEEDS = [48, 49, 50]
ARMS = ['adaptive', 'control']
COLORS = {'adaptive': '#3366a0', 'control': '#b76d33'}
LABELS = {'adaptive': 'KL 反馈', 'control': '固定计划'}
REPEATS = ['重复一', '重复二', '重复三']
STYLES = ['-', '--', ':']
DIAGNOSTICS = [
    'actor_lr_multiplier', 'actor_lr_multiplier_next', 'actor_learning_rate',
    'critic_learning_rate', 'policy_kl_after_update', 'kl_ema',
    'adaptive_kl_target', 'policy_clip_fraction_after_update', 'clip_range',
]


def checkpoint_record(selection):
    return selection['median_best' if selection['selected_kind']=='stable_model' else 'mean_best']


def bootstrap_pair(delta):
    rng = np.random.default_rng(20261022)
    sampled = delta[rng.integers(0, len(delta), (20000, len(delta)))].mean(axis=1)
    return np.percentile(sampled, [2.5, 97.5]).tolist()


def two_way_interval(delta):
    rng = np.random.default_rng(20261022)
    runs = rng.integers(delta.shape[0], size=(20000, delta.shape[0]))
    games = rng.integers(delta.shape[1], size=(20000, delta.shape[1]))
    samples = delta[runs[:, :, None], games[:, None, :]].mean(axis=(1, 2))
    return np.percentile(samples, [2.5, 97.5]).tolist()


def collect():
    DATA.mkdir(parents=True, exist_ok=True)
    sources = {}
    def register(path):
        path = Path(path)
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    def read_json(path):
        return json.loads(register(path).read_text())
    def scores(path, episodes):
        path = register(path)
        frame = pd.read_csv(path).sort_values('seed').reset_index(drop=True)
        if len(frame)!=episodes or frame.seed.duplicated().any() or not np.isfinite(frame.reward).all():
            raise ValueError(f'Invalid complete score file: {path}')
        meta = read_json(path.with_suffix('.json'))
        return frame, meta

    output = ROOT/'results/evaluation'/STUDY
    protocol = read_json(output/'protocol.json')
    environment = protocol['environment']
    curves, selections, diagnostics, test_rows, paired_rows = [], [], [], [], []
    common_keys = [
        'n_envs', 'n_steps', 'batch_size', 'n_epochs', 'learning_rate', 'final_learning_rate',
        'gamma', 'gae_lambda', 'clip_range', 'final_clip_range', 'ent_coef', 'final_ent_coef',
        'vf_coef', 'target_kl', 'critic_max_grad_norm', 'huber_delta', 'value_loss',
        'critic_lr_multiplier', 'extra_critic_epochs', 'adaptive_kl_target', 'adaptive_kl_band',
        'adaptive_kl_factor', 'adaptive_kl_alpha', 'actor_lr_min_multiplier',
        'actor_lr_max_multiplier', 'eval_seed', 'n_eval_episodes', 'eval_envs',
        'checkpoint_selection', 'selection_mean_tolerance', 'resume',
    ]
    common = None
    kl_scores = {arm: [] for arm in ARMS}
    for repeat, seed in enumerate(SEEDS, 1):
        for arm in ARMS:
            run = f'{STUDY}_{arm}_s{seed}'
            base = ROOT/'logs'/run
            config = read_json(base/'config.json')
            args = config['arguments']
            settings = {key: args[key] for key in common_keys}
            if common is None:
                common = settings
            elif settings!=common:
                raise ValueError(f'Unmatched experimental settings: {run}')
            if config['environment']!=environment or args['seed']!=seed:
                raise ValueError('Run environment/seed mismatch')
            if args['adaptive_kl_enabled']!=(arm=='adaptive'):
                raise ValueError('Unexpected feedback arm')
            if not args['fixed_eval_seeds'] or args['post_eval_seed']!=141000:
                raise ValueError('Unexpected development/test seed protocol')
            training = read_json(base/'training_summary.json')
            if training['status']!='completed' or training['stage_timesteps']!=503808:
                raise ValueError(f'Training not complete: {run}')
            selection = read_json(base/'selected_model.json')
            chosen = checkpoint_record(selection)
            with np.load(register(base/'eval/evaluations.npz')) as archive:
                steps, rewards = archive['timesteps'], archive['results']
                if rewards.shape!=(len(steps),32) or not np.isfinite(rewards).all():
                    raise ValueError('Expected complete 32-game development evaluations')
                if steps[-1]!=503808 or len(np.unique(steps))!=len(steps):
                    raise ValueError('Missing/duplicate final development evaluation')
                for step, values in zip(steps, rewards):
                    episodes = pd.read_csv(register(base/f'eval/episodes_{int(step)}.csv')).sort_values('seed')
                    np.testing.assert_array_equal(episodes.seed,np.arange(131000,131032))
                    np.testing.assert_array_equal(np.sort(values),np.sort(episodes.reward))
                    curves.append(dict(run=run, seed=seed, repeat=repeat, arm=arm, step=int(step),
                        episodes=len(values), mean=float(values.mean()), median=float(np.median(values)),
                        p25=float(np.percentile(values,25)), below50=int(np.sum(values<50))))
                index = np.flatnonzero(steps==chosen['timesteps'])
                if len(index)!=1 or not np.isclose(rewards[index[0]].mean(),chosen['mean_reward']):
                    raise ValueError('Selected model does not match development evidence')
            selections.append(dict(run=run, seed=seed, repeat=repeat, arm=arm,
                selected_kind=selection['selected_kind'], selected_step=chosen['timesteps'],
                selected_mean=chosen['mean_reward'], selected_median=chosen['median_reward'],
                mean_best_step=selection['mean_best']['timesteps'],
                mean_best_reward=selection['mean_best']['mean_reward']))
            progress = pd.read_csv(register(base/'progress.csv'))
            wanted = ['time/total_timesteps']+['train/'+key for key in DIAGNOSTICS]
            progress = progress[wanted].dropna()
            if len(progress)<100 or not np.isfinite(progress.to_numpy()).all():
                raise ValueError('Incomplete optimization diagnostics')
            progress = progress.rename(columns={'time/total_timesteps':'logged_step',
                **{'train/'+key:key for key in DIAGNOSTICS}})
            for row in progress.to_dict('records'):
                diagnostics.append(dict(run=run, seed=seed, repeat=repeat, arm=arm, **row))
            post = read_json(base/'post_evaluation.json')
            if post['status']!='completed':
                raise ValueError('Selected-model test incomplete')
            frame, meta = scores(output/run/'ppo_results.csv',50)
            np.testing.assert_array_equal(frame.seed, np.arange(141000,141050))
            if meta['environment']!=environment or meta['model']!=selection['model']:
                raise ValueError('Test environment/selected-model mismatch')
            kl_scores[arm].append(frame.reward.to_numpy(float))
            test_rows.append(dict(run=run, repeat=repeat, arm=arm,
                mean=float(frame.reward.mean()), median=float(frame.reward.median()),
                below50=int((frame.reward<50).sum()),
                truncated=int(frame.truncated.astype(str).str.lower().eq('true').sum())))
    curves = pd.DataFrame(curves)
    selections = pd.DataFrame(selections)
    diagnostics = pd.DataFrame(diagnostics)
    test_rows = pd.DataFrame(test_rows)

    historical = [
        ('tricks','稳定续训组合 − 仅续训','30 局，单次训练',
         'ppo_optimization_20261022/refine_heldout/ppo_results.csv',
         'ppo_optimization_20261022/control_heldout/ppo_results.csv',30),
        ('shaping','辅助奖励 − 原始奖励','50 局，单次训练',
         'ppo_long_horizon_20261022/shaped_heldout/ppo_results.csv',
         'ppo_long_horizon_20261022/native_heldout/ppo_results.csv',50),
        ('balanced','独立裁剪+Huber − 标准更新','50 局，单次训练',
         'ppo_balanced_20261022/ppo_custom_fixed_angle_balanced_500k_20261022/ppo_results.csv',
         'ppo_balanced_20261022/ppo_custom_fixed_angle_balanced_control_500k_20261022/ppo_results.csv',50),
        ('dual_rate','双学习率+额外拟合 − 对照','50 局，单次训练，含早停',
         'ppo_dual_rate_20261022/ppo_custom_fixed_angle_dual_rate_500k_20261022/ppo_results.csv',
         'ppo_dual_rate_20261022/ppo_custom_fixed_angle_dual_rate_control_500k_20261022/ppo_results.csv',50),
        ('gae','GAE 0.99 − 0.98','50 局，单次训练',
         'ppo_gae_20261022/ppo_custom_fixed_angle_gae_500k_20261022/ppo_results.csv',
         'ppo_gae_20261022/ppo_custom_fixed_angle_gae_control_500k_20261022/ppo_results.csv',50),
    ]
    published = {
        'tricks': ('results/evaluation/ppo_optimization_20261022/summary.json',
                   ['paired_heldout_effects','control'],'mean_difference','paired_bootstrap_ci95'),
        'balanced': ('results/evaluation/ppo_balanced_20261022/comparison_summary.json',
                     ['paired_comparisons','balanced_minus_control'],'mean_difference','bootstrap_95_percent_ci'),
        'dual_rate': ('results/evaluation/ppo_dual_rate_20261022/comparison_summary.json',
                      ['paired','dual_rate_minus_control'],'mean_difference','bootstrap_95_ci'),
        'gae': ('report/data/ppo_report_summary.json',
                ['paired','ppo99_minus_ppo98'],'mean','ci95'),
    }
    effects = []
    for key, label, note, improved, reference, episodes in historical:
        a, ma = scores(ROOT/'results/evaluation'/improved, episodes)
        b, mb = scores(ROOT/'results/evaluation'/reference, episodes)
        if ma['environment']!=environment or mb['environment']!=environment:
            raise ValueError('Historical environment mismatch')
        np.testing.assert_array_equal(a.seed,b.seed)
        delta = a.reward.to_numpy(float)-b.reward.to_numpy(float)
        interval_source = 'new_paired_bootstrap'
        if key in published:
            filename, keys, mean_key, ci_key = published[key]
            record = read_json(ROOT/filename)
            for field in keys:
                record = record[field]
            if not np.isclose(record[mean_key],delta.mean()):
                raise ValueError('Published effect does not match raw paired scores')
            low, high = record[ci_key]
            interval_source = filename
        else:
            low, high = bootstrap_pair(delta)
        effects.append(dict(key=key, label=label, note=note, mean=float(delta.mean()),
            ci_low=low, ci_high=high, episodes=episodes, training_repeats=1,
            interval='paired_episode_bootstrap', interval_source=interval_source,
            wins=int(np.sum(delta>0)),
            ties=int(np.sum(delta==0)), losses=int(np.sum(delta<0))))
        for game_seed, x, y in zip(a.seed,a.reward,b.reward):
            paired_rows.append(dict(experiment=key, repeat=1, game_seed=int(game_seed),
                improved=x, control=y, difference=x-y))
    delta = np.asarray(kl_scores['adaptive'])-np.asarray(kl_scores['control'])
    low, high = two_way_interval(delta)
    official = read_json(output/'comparison_summary.json')
    if official['status']!='completed':
        raise ValueError('Study comparison is incomplete')
    published_kl = official['paired_adaptive_minus_control']
    np.testing.assert_allclose([delta.mean(),low,high],
        [published_kl['mean_difference'],*published_kl['ci95']],rtol=0,atol=1e-10)
    effects.append(dict(key='adaptive_kl',label='KL 反馈 − 固定计划',note='每次 50 局，三次配对续训',
        mean=float(delta.mean()),ci_low=low,ci_high=high,episodes=50,training_repeats=3,
        interval='paired_two_way_bootstrap',interval_source='new_paired_bootstrap',
        wins=int(np.sum(delta>0)),
        ties=int(np.sum(delta==0)),losses=int(np.sum(delta<0))))
    for repeat, values in enumerate(delta,1):
        for game_seed, value, a, b in zip(range(141000,141050), values,
                kl_scores['adaptive'][repeat-1],kl_scores['control'][repeat-1]):
            paired_rows.append(dict(experiment='adaptive_kl',repeat=repeat,game_seed=game_seed,
                improved=a,control=b,difference=value))
    effects = pd.DataFrame(effects)
    summary = dict(created_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),study=STUDY,
        training_seeds=SEEDS,development_episodes=32,test_episodes_per_run=50,
        actual_added_steps_per_run=503808,bootstrap_replicates=20000,bootstrap_seed=20261022,
        bootstrap_scope='20,000 draws and the stated RNG seed apply to newly computed shaping and KL intervals. Other historical intervals are retained from their original summaries after checking raw paired means.',
        development_seed_range=[131000,131031],test_seed_range=[141000,141049],
        shared_settings=common,source_sha256=protocol['source_sha256'],sources_sha256=sources,
        kl_per_repeat_effect=delta.mean(axis=1).tolist(),
        kl_mean_scores={arm:float(np.asarray(kl_scores[arm]).mean()) for arm in ARMS},
        interval_scope='Historical intervals condition on fixed policies. KL uses paired two-way resampling of 3 continuation seeds and shared episode seeds; not independent from-scratch training.',
        diagnostics_scope='Metrics use their CSV output step; output can lag its optimization rollout. Clip fraction is abs(ratio-1)>epsilon, not the fraction of policy gradients clipped.',
        historical_scope='Each row uses its own same-game-seed control. Composite changes are not component-level ablations; dual-rate stopped earlier and retained its source checkpoint.',
        evaluation_scope='KL continuation tests use a different set of game seeds from the existing DQN/GAE comparison; no cross-round raw-score ranking.',
        frozen_file_policy='Original training code and configurations are not modified by this report script.')
    derived = {}
    for name, frame in [('development_curves',curves),('checkpoint_selection',selections),
                        ('kl_diagnostics',diagnostics),('ablation_effects',effects),
                        ('ablation_pairs',pd.DataFrame(paired_rows)),('kl_test_summary',test_rows)]:
        path = DATA/f'{name}.csv'
        frame.to_csv(path,index=False)
        derived[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary['derived_sha256'] = derived
    (DATA/'analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return curves,selections,diagnostics,effects,test_rows,summary


def load_snapshot():
    summary = json.loads((DATA/'analysis_summary.json').read_text())
    for name,digest in summary['derived_sha256'].items():
        if hashlib.sha256((DATA/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Derived data snapshot changed: {name}')
    frames = [pd.read_csv(DATA/f'{name}.csv',float_precision='round_trip') for name in [
        'development_curves','checkpoint_selection','kl_diagnostics',
        'ablation_effects','kl_test_summary']]
    return *frames, summary


def learning_curves(curves, selections):
    fig, axes = plt.subplots(2,3,figsize=(10,6.5),sharex=True,sharey='row')
    for col in range(3):
        for arm in ARMS:
            data = curves[(curves.repeat==col+1)&(curves.arm==arm)]
            selected = selections[(selections.repeat==col+1)&(selections.arm==arm)].iloc[0]
            for row, metric in enumerate(['mean','median']):
                ax = axes[row,col]
                ax.plot(data.step/10000,data[metric],'-o',color=COLORS[arm],
                        label=LABELS[arm],lw=1.5,ms=3.5)
                point = data[data.step==selected.selected_step].iloc[0]
                ax.scatter(point.step/10000,point[metric],marker='*',s=125,
                    color=COLORS[arm],edgecolors='white',linewidths=.5,zorder=5)
                if row==0:
                    best = data[data.step==selected.mean_best_step].iloc[0]
                    ax.scatter(best.step/10000,best['mean'],marker='D',s=62,
                        facecolors='none',edgecolors=COLORS[arm],linewidths=1.2,zorder=6)
                ax.grid(alpha=.18)
                ax.set_xlim(-1,52)
                ax.set_ylim(0,275)
        axes[0,col].set_title(REPEATS[col],fontproperties=PANEL_FONT)
        axes[1,col].set_xlabel('本阶段新增决策步 / 万步')
    axes[0,0].set_ylabel('32 局开发均分')
    axes[1,0].set_ylabel('32 局开发中位数')
    legend = [Line2D([],[],color=COLORS[a],lw=1.8,label=LABELS[a]) for a in ARMS]
    legend += [Line2D([],[],color='#475569',marker='*',ls='',ms=10,label='最终选中检查点'),
               Line2D([],[],color='#475569',marker='D',mfc='none',ls='',ms=6,label='均分最佳节点')]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.945),ncol=4,frameon=False,fontsize=9)
    fig.suptitle('PPO 多次配对续训：开发表现与检查点选择',fontproperties=TITLE_FONT)
    fig.tight_layout(rect=(0,0,1,.88))
    save(fig,'ppo_multiseed_curves.png')


def ablations(effects, summary):
    fig,ax = plt.subplots(figsize=(10,5.7))
    for i,row in effects.iterrows():
        current = row.key=='adaptive_kl'
        color = '#267f70' if current else '#3366a0'
        y = len(effects)-1-i
        if current:
            ax.axhspan(y-.43,y+.43,color=color,alpha=.07)
            ax.scatter(summary['kl_per_repeat_effect'],[y-.16,y,y+.16],
                color='#8793a2',s=25,zorder=4)
        ax.errorbar(row['mean'],y,xerr=[[row['mean']-row.ci_low],[row.ci_high-row['mean']]],
            fmt='D' if current else 'o',color=color,ecolor=color,capsize=4,
            markersize=7 if current else 6,lw=1.8,zorder=5)
        ax.text(1.035,y,f'{row["mean"]:+.2f} [{row.ci_low:.2f}, {row.ci_high:.2f}]',
            transform=ax.get_yaxis_transform(),va='center',fontsize=9.5,color=color)
    ax.set_yticks(range(len(effects)-1,-1,-1),
        [f'{row.label}\n{row.note}' for row in effects.itertuples()],fontsize=10)
    ax.axvline(0,color='#64748b',ls='--',lw=1.2)
    ax.set(xlabel='改动 − 同轮对照的配对均分差 / 分',
        xlim=(min(-65,effects.ci_low.min()-8),max(80,effects.ci_high.max()+8)),
        ylim=(-.65,len(effects)-.35))
    ax.grid(axis='x',alpha=.18)
    ax.text(1.035,1.03,'均分差 [95% 区间]',transform=ax.transAxes,fontsize=10)
    ax.legend(handles=[
        Line2D([],[],marker='o',color='#3366a0',ls='-',label='固定策略：回合配对区间'),
        Line2D([],[],marker='D',color='#267f70',ls='-',label='KL：训练/游戏两维配对区间'),
        Line2D([],[],marker='o',color='#8793a2',ls='',label='KL 单次续训均分差')],
        loc='upper center',bbox_to_anchor=(.55,1.18),ncol=1,frameon=False,fontsize=9)
    fig.suptitle('PPO 改进消融：保留同轮对照与负结果',fontproperties=TITLE_FONT)
    fig.subplots_adjust(left=.32,right=.73,bottom=.14,top=.77)
    save(fig,'ppo_ablation_effects.png')


def controller_diagnostics(data):
    fig,axes = plt.subplots(2,3,figsize=(10,6.5),sharex=True)
    panels = [
        ('actor_lr_multiplier','策略学习率倍率','相对基础计划的倍率'),
        ('actor_learning_rate','实际策略学习率','学习率'),
        ('policy_kl_after_update','整轮更新后的采样 KL','近似 KL'),
        ('kl_ema','平滑 KL 与反馈阈值','KL 的 EMA'),
        ('policy_clip_fraction_after_update','整轮概率比越界比例','越界样本比例'),
        ('clip_range','概率比裁剪范围退火','裁剪范围 ε'),
    ]
    for index,(metric,title,ylabel) in enumerate(panels):
        ax=axes.flat[index]
        for repeat in range(1,4):
            for arm in ARMS:
                # All six runs have the same clip schedule; show it once.
                if metric=='clip_range' and (repeat!=1 or arm!='control'):continue
                if metric in ['actor_lr_multiplier','kl_ema'] and arm=='control':continue
                frame=data[(data.repeat==repeat)&(data.arm==arm)]
                ax.plot(frame.logged_step/10000,frame[metric],ls=STYLES[repeat-1],
                    color='#64748b' if metric=='clip_range' else COLORS[arm],
                    lw=1.3,alpha=.8)
        ax.set_title(title,fontproperties=PANEL_FONT)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.18)
        ax.set_xlim(0,51)
    axes[0,0].axhline(1,color=COLORS['control'],lw=1.2)
    axes[0,0].set_ylim(.45,2.12)
    axes[0,1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axes[0,2].set_yscale('log')
    axes[0,2].axhline(.0005,color='#475569',ls='-.',lw=1.1,label='反馈目标')
    axes[0,2].legend(frameon=False,fontsize=9,loc='lower left')
    axes[1,0].axhspan(.0005/1.5,.0005*1.5,color='#94a3b8',alpha=.16)
    axes[1,0].axhline(.0005,color='#475569',ls='-.',lw=1)
    axes[1,0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axes[1,1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes[1]:ax.set_xlabel('本阶段日志记录步数 / 万步')
    handles=[Line2D([],[],color=COLORS[a],lw=1.7,label=LABELS[a]) for a in ARMS]
    handles += [Line2D([],[],color='#64748b',ls=style,lw=1.5,label=label)
                for style,label in zip(STYLES,REPEATS)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.945),
               ncol=5,frameon=False,fontsize=9)
    fig.suptitle('KL 反馈诊断：学习率、策略变化与裁剪调度',fontproperties=TITLE_FONT)
    fig.tight_layout(rect=(0,0,1,.88))
    save(fig,'ppo_kl_diagnostics.png')


def write_tex(effects, tests, summary):
    lines=['% Generated by report/generate_ppo_analysis.py from measured data.']
    def number(name,value):
        lines.append('\\newcommand{\\'+name+'}{'+f'{value:.2f}'+'}')
    for row in effects.itertuples():
        name={'tricks':'Tricks','shaping':'Shaping','balanced':'Balanced',
              'dual_rate':'DualRate','gae':'Gae','adaptive_kl':'KL'}[row.key]
        number(name+'AblationDiff',row.mean)
        number(name+'AblationLow',row.ci_low)
        number(name+'AblationHigh',row.ci_high)
    for i,word in enumerate(['One','Two','Three']):
        number('KLRepeat'+word+'Diff',summary['kl_per_repeat_effect'][i])
        for arm,label in [('adaptive','Adaptive'),('control','Control')]:
            record=tests[(tests.repeat==i+1)&(tests.arm==arm)].iloc[0]
            number('KLRepeat'+word+label+'Mean',record['mean'])
    number('KLAdaptiveMean',summary['kl_mean_scores']['adaptive'])
    number('KLControlMean',summary['kl_mean_scores']['control'])
    (ROOT/'report/latex/ppo_analysis_metrics.tex').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-snapshot',action='store_true')
    args=parser.parse_args()
    configure()
    curves,selections,diagnostics,effects,tests,summary=load_snapshot() if args.from_snapshot else collect()
    learning_curves(curves,selections)
    ablations(effects,summary)
    controller_diagnostics(diagnostics)
    write_tex(effects,tests,summary)
    print(effects[['key','mean','ci_low','ci_high']].to_string(index=False))
    print('KL test means:',summary['kl_mean_scores'])
    print('KL per-repeat effects:',summary['kl_per_repeat_effect'])


if __name__=='__main__':main()
