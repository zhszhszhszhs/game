"""Rebuild report figures from saved logs and real, deterministic model rollouts.

Run with the project interpreter. The paired CSV files are produced separately
by src/evaluate.py; no training or synthetic episode data is used here.
"""
from collections import deque
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np
import pandas as pd
from PIL import Image
import torch
from stable_baselines3 import DQN

from env import environment_spec, make_env

IMAGES = ROOT / 'report/latex/images'
DATA = ROOT / 'report/data'
TITLE_FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', size=16)
PANEL_FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', size=11)
MODEL = ROOT / 'models/dqn_custom_fixed_angle_refine_200k/best_model/best_model.zip'
STAGES = [
    ('500k', '从头训练', '#3366a0', 20),
    ('refine_200k', '低学习率续训', '#267f70', 50),
    ('recovery_100k', '三步回报恢复', '#a96633', 50),
    ('retention_100k', '一步扩大回放', '#8a5e9e', 50),
]


def configure():
    IMAGES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    fontManager.addfont(font)
    fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
    plt.rcParams.update({
        'font.family': FontProperties(fname=font).get_name(),
        'font.size': 11, 'axes.unicode_minus': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelcolor': '#334155', 'text.color': '#182c44',
        'xtick.color': '#475569', 'ytick.color': '#475569',
        'savefig.facecolor': 'white',
        'axes.titlesize': 11, 'figure.titlesize': 16,
    })


def save(fig, filename):
    # Equal canvas widths keep title sizes equal after inclusion in the report.
    assert fig.get_figwidth() == 10
    assert all(t is fig._suptitle for t in fig.texts), 'Report images must have no footer notes'
    fig.savefig(IMAGES / filename, dpi=240)
    plt.close(fig)
    print(f'Saved {filename}', flush=True)


def training_curves():
    fig, axes = plt.subplots(2, 4, figsize=(10, 6.5), sharey='row')
    evidence = []
    for col, (suffix, title, color, n) in enumerate(STAGES):
        path = ROOT / 'logs' / f'dqn_custom_fixed_angle_{suffix}'
        monitor = pd.read_csv(path / 'train.monitor.csv', comment='#')
        evaluation = np.load(path / 'eval/evaluations.npz')
        steps = monitor.l.cumsum() / 1000
        x = evaluation['timesteps'] / 1000
        means = evaluation['results'].mean(axis=1)
        ax = axes[0, col]
        ax.scatter(steps, monitor.r, s=4, alpha=.18, color=color, rasterized=True)
        ax.plot(steps, monitor.r.rolling(20, min_periods=1).mean(), color=color,
                lw=1.6, label='20 回合滚动均值')
        ax.set_title(title, fontproperties=PANEL_FONT, pad=12)
        ax.set_ylim(0, 620)
        ax.grid(axis='y', alpha=.18)
        if col == 0:
            ax.set_ylabel('训练回合得分')
            ax.legend(fontsize=9, loc='upper left', frameon=False)
        if suffix != '500k':
            ax.axvspan(0, 20, color='#64748b', alpha=.08)
        ax = axes[1, col]
        ax.plot(x, means, '-o', color=color, lw=1.8, ms=4)
        best = int(means.argmax())
        ax.scatter(x[best], means[best], s=95, marker='*', color='#c54d35', zorder=4)
        ax.annotate(f'{means[best]:.2f}', (x[best], means[best]),
                    xytext=(5, 10), textcoords='offset points', fontsize=10, color='#a63d2c')
        ax.annotate(f'{means[-1]:.2f}', (x[-1], means[-1]),
                    xytext=(-4, -16), textcoords='offset points', ha='right', fontsize=10)
        if suffix in ('recovery_100k', 'retention_100k'):
            ax.axvline(x[-1], color=color, ls=':', alpha=.7)
            ax.text(.97, .04, '50k 早停', transform=ax.transAxes, ha='right', fontsize=9)
        ax.text(.98, .92, f'每节点 {n} 局', transform=ax.transAxes, ha='right', fontsize=9)
        ax.set_ylim(0, 340)
        ax.set_xlabel('本阶段决策步 / 千步')
        ax.grid(axis='y', alpha=.18)
        if col == 0:
            ax.set_ylabel('贪心开发评估均分')
        for row in range(2):
            axes[row, col].set_xlim(0, x[-1] * 1.08)
        evidence.append(dict(run=path.name, training_episodes=len(monitor),
                             evaluation_steps=evaluation['timesteps'].tolist(),
                             evaluation_means=means.tolist()))
    fig.suptitle('训练奖励与开发评估：保留全部原始采样点', fontproperties=TITLE_FONT)
    fig.tight_layout(rect=(0, 0, 1, .95), w_pad=1)
    save(fig, 'training_curves.png')
    (DATA / 'training_sources.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')


def distribution():
    best = pd.read_csv(ROOT / 'results/evaluation/report_best_30/dqn_results.csv')
    final = pd.read_csv(ROOT / 'results/evaluation/report_final_30/dqn_results.csv')
    assert best.seed.tolist() == final.seed.tolist() == list(range(42000, 42030))
    paired = best[['seed', 'reward']].rename(columns={'reward': 'best_reward'})
    paired['final_reward'] = final.reward
    paired['difference'] = best.reward - final.reward
    paired.to_csv(DATA / 'paired_scores.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={'width_ratios': [1.8, 1]})
    ax = axes[0]
    for i, row in paired.iterrows():
        ax.plot([i, i], [row.final_reward, row.best_reward], color='#b8c4cf', lw=1, zorder=1)
    ax.scatter(range(30), final.reward, c='#9b6c56', marker='s', s=27, label='续训最终模型', zorder=2)
    ax.scatter(range(30), best.reward, c='#267f70', s=32, label='续训最佳模型', zorder=3)
    ax.set_xticks(range(0, 30, 5), [str(x) for x in range(42000, 42030, 5)])
    ax.set_xlabel('逐局开发种子（共 30 局）')
    ax.set_ylabel('整局原生得分')
    ax.set_title('相同开局种子的配对结果', fontproperties=PANEL_FONT)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis='y', alpha=.18)
    ax = axes[1]
    box = ax.boxplot([final.reward, best.reward], tick_labels=['续训最终', '续训最佳'],
                     patch_artist=True, showfliers=False, widths=.45,
                     medianprops={'color': '#172b43', 'linewidth': 2})
    for patch, color in zip(box['boxes'], ['#d6baab', '#a5cec5']):
        patch.set_facecolor(color)
    offsets = np.linspace(-.12, .12, 30)
    for i, (scores, color) in enumerate([(final.reward, '#9b6c56'), (best.reward, '#267f70')], 1):
        ax.scatter(i + offsets, scores, s=16, alpha=.65, color=color)
        ax.scatter(i, scores.mean(), marker='D', s=35, color='#c54d35', zorder=4)
        ax.text(i, 570, f'均值 {scores.mean():.2f}\n中位数 {scores.median():.2f}',
                ha='center', va='top', fontsize=10)
    ax.set_ylim(0, 620)
    ax.set_title('分布、均值与中位数', fontproperties=PANEL_FONT)
    ax.grid(axis='y', alpha=.18)
    wins = int((paired.difference > 0).sum())
    fig.suptitle('30 个开发种子上的模型复测', fontproperties=TITLE_FONT)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, 'score_distribution.png')
    summary = dict(seed_start=42000, episodes=30, best_mean=float(best.reward.mean()),
                   best_median=float(best.reward.median()), final_mean=float(final.reward.mean()),
                   final_median=float(final.reward.median()), best_wins=wins,
                   mean_paired_difference=float(paired.difference.mean()),
                   purpose='development_diagnosis_not_final_test')
    (DATA / 'paired_summary.json').write_text(json.dumps(summary, indent=2)+'\n')


def diagnostics():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
    sources = []
    for col, (suffix, title, color, _) in enumerate(STAGES[:2]):
        path = ROOT / 'logs' / f'dqn_custom_fixed_angle_{suffix}'
        progress = pd.read_csv(path / 'progress.csv')
        evaluation = np.load(path / 'eval/evaluations.npz')
        best_step = float(evaluation['timesteps'][evaluation['results'].mean(axis=1).argmax()]) / 1000
        exploration = progress[['time/total_timesteps', 'rollout/exploration_rate']].dropna()
        loss = progress[['time/total_timesteps', 'train/loss']].dropna()
        assert (loss['train/loss'] > 0).all()
        exploration.to_csv(DATA / f'exploration_{suffix}.csv', index=False)
        loss.to_csv(DATA / f'loss_{suffix}.csv', index=False)
        ax = axes[0, col]
        ax.plot(exploration['time/total_timesteps'] / 1000,
                exploration['rollout/exploration_rate'], color=color, lw=1.8)
        ax.set_title(title, fontproperties=PANEL_FONT)
        ax.set_ylabel('探索概率 ε')
        ax.set_ylim(0, 1.05 if col == 0 else .055)
        ax = axes[1, col]
        x = loss['time/total_timesteps'] / 1000
        ax.scatter(x, loss['train/loss'], s=10, color=color, alpha=.25, label='原始日志点')
        ax.plot(x, loss['train/loss'].rolling(20, min_periods=1).median(), color=color,
                lw=1.6, label='20 个日志点滚动中位数')
        ax.set_yscale('log')
        ax.set_ylabel('Huber 损失（对数轴）')
        ax.set_xlabel('本阶段决策步 / 千步')
        ax.legend(fontsize=8.5, frameon=False, loc='lower right')
        for row in range(2):
            axes[row, col].axvspan(0, 20, color='#64748b', alpha=.08)
            axes[row, col].axvline(best_step, color='#c54d35', ls='--', lw=1)
            axes[row, col].grid(axis='y', alpha=.18)
        axes[0, col].text(.98, .9, f'开发最佳：{best_step:g}k', ha='right',
                          transform=axes[0, col].transAxes, fontsize=10, color='#a63d2c')
        sources.append(dict(run=path.name, exploration_points=len(exploration), loss_points=len(loss),
                            best_step=int(best_step * 1000)))
    fig.suptitle('探索率与 TD 学习信号的实际变化', fontproperties=TITLE_FONT)
    fig.tight_layout(rect=(0, 0, 1, .95), h_pad=1.4)
    save(fig, 'training_diagnostics.png')
    (DATA / 'diagnostic_sources.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2)+'\n')


def robustness():
    best = pd.read_csv(ROOT / 'results/evaluation/report_best_30/dqn_results.csv')
    final = pd.read_csv(ROOT / 'results/evaluation/report_final_30/dqn_results.csv')
    assert best.seed.tolist() == final.seed.tolist() == list(range(42000, 42030))
    paired = pd.DataFrame({'seed': best.seed, 'difference': best.reward - final.reward})
    rng = np.random.default_rng(20261017)
    indices = rng.integers(0, len(paired), size=(20000, len(paired)))
    bootstrap = paired.difference.to_numpy()[indices].mean(axis=1)
    ci = np.quantile(bootstrap, [.025, .975])
    statistics = []
    for label, data in [('最佳模型', best), ('续训最终模型', final)]:
        scores = data.reward
        statistics.append(dict(model=label, mean=float(scores.mean()), std=float(scores.std(ddof=0)),
                               median=float(scores.median()), p10=float(scores.quantile(.1)),
                               p25=float(scores.quantile(.25)), below_50=int((scores < 50).sum()),
                               below_50_fraction=float((scores < 50).mean()),
                               min=float(scores.min()), max=float(scores.max()),
                               timeout_count=int(data.truncated.sum()),
                               full_clear_count=int((scores == 864).sum()),
                               mean_episode_length=float(data.length.mean())))
    pd.DataFrame(statistics).to_csv(DATA / 'robustness_statistics.csv', index=False)
    summary = dict(episodes=30, statistics=statistics, bootstrap_seed=20261017,
                   bootstrap_repetitions=20000, bootstrap_unit='paired_episode_seed',
                   mean_paired_difference=float(paired.difference.mean()), paired_bootstrap_95pct_ci=ci.tolist(),
                   best_wins=int((paired.difference > 0).sum()),
                   losing_seeds=paired.loc[paired.difference < 0].to_dict(orient='records'),
                   purpose='conditional_development_diagnosis_not_final_test')
    (DATA / 'robustness_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9))
    ax = axes[0]
    for label, data, color in [('续训最终', final, '#9b6c56'), ('续训最佳', best, '#267f70')]:
        ordered = np.sort(data.reward.to_numpy())
        ax.step(np.r_[0, ordered], np.r_[0, np.arange(1, 31) / 30], where='post',
                color=color, lw=1.8, label=label)
    ax.axvline(50, color='#64748b', ls=':', lw=1)
    ax.text(58, .33, '低于 50 分：9/30', color='#9b6c56', fontsize=10)
    ax.text(58, .07, '低于 50 分：1/30', color='#267f70', fontsize=10)
    ax.set_xlabel('整局得分阈值')
    ax.set_ylabel('得分不超过阈值的局数比例')
    ax.set_title('经验累积分布：观察低分尾部', fontproperties=PANEL_FONT)
    ax.set_ylim(0, 1.04)
    ax.set_xlim(0, 470)
    ax.legend(frameon=False, fontsize=10, loc='lower right')
    ax.grid(alpha=.18)
    ordered = paired.sort_values('difference').reset_index(drop=True)
    ax = axes[1]
    colors = np.where(ordered.difference > 0, '#267f70', '#a6523b')
    ax.bar(np.arange(1, 31), ordered.difference, color=colors, width=.8)
    ax.axhline(0, color='#475569', lw=1)
    ax.axhline(paired.difference.mean(), color='#c54d35', ls='--', lw=1,
               label=f'均值 {paired.difference.mean():.2f} 分')
    for rank, row in ordered.head(2).iterrows():
        ax.annotate(f'种子 {int(row.seed)}\n{row.difference:g} 分', (rank+1, row.difference),
                    xytext=(12 if rank == 0 else 16, -15 if rank == 0 else -30),
                    textcoords='offset points', fontsize=8.5)
    ax.set_ylim(-290, 440)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel('配对差的排序位置（从小到大）')
    ax.set_ylabel('最佳模型得分 − 最终模型得分')
    ax.set_title('逐种子收益与局部退步', fontproperties=PANEL_FONT)
    ax.legend(frameon=False, fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=.18)
    fig.suptitle('平均提升之外：低分风险与配对差', fontproperties=TITLE_FONT)
    fig.tight_layout(rect=(0, 0, 1, .94), w_pad=1.6)
    save(fig, 'score_robustness.png')


def rollout_figures():
    torch.set_num_threads(1)
    model = DQN.load(MODEL, device='cpu')
    environment_spec(model)
    history = deque(maxlen=80)
    successes, failures = [], []
    pending = None
    raw = None
    last = None
    pipeline = None

    def observer(frame, reward, lives):
        nonlocal pending, last
        if frame is None:
            history.clear()
            last = None
            return
        item = dict(frame=raw.frames, image=frame.copy(), score=raw.score, lives=raw.lives,
                    ball=raw.ball.copy().tolist(), velocity=raw.velocity.copy().tolist(),
                    paddle_x=raw.paddle_x)
        history.append(item)
        if pending and raw.frames == pending[0]:
            successes.append(pending[1] + [item])
            pending = None
        if last and len(history) >= 38:
            if not successes and pending is None and raw.score > 0 and last['velocity'][1] > 0 > raw.velocity[1] and last['ball'][1] > 175:
                pending = (raw.frames + 12, [history[-25], history[-13], item])
            if not failures and lives < last['lives']:
                failures.append([history[-37], history[-25], history[-13], item])
        last = item

    env = make_env(42000, render_mode='rgb_array', frame_callback=observer)
    raw = env.venv.venv.envs[0].unwrapped
    obs = env.reset()
    try:
        for step in range(6000):
            action = model.predict(obs, deterministic=True)[0]
            obs, _, done, _ = env.step(action)
            if pipeline is None and step >= 20 and raw.velocity[1] > 0 and raw.ball[1] > 145 and len(history) >= 4:
                last_two = [history[-2]['image'], history[-1]['image']]
                pooled = np.maximum(*last_two)
                gray = cv2.resize(cv2.cvtColor(pooled, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA)
                assert np.array_equal(gray, obs[0, -1]), 'Plot must match actual network input'
                pipeline = dict(raw=last_two[-1], pooled=pooled, stack=obs[0].copy(), step=step+1,
                                frame=raw.frames)
            if pipeline is not None and successes and failures:
                break
            if done[0]:
                raise RuntimeError('Requested behavior events were not found in this episode')
    finally:
        env.close()
    if not (pipeline and successes and failures):
        raise RuntimeError('Incomplete rollout evidence')
    np.savez_compressed(DATA / 'observation_samples.npz', raw=pipeline['raw'], pooled=pipeline['pooled'],
                        stack=pipeline['stack'])
    fig, axes = plt.subplots(1, 6, figsize=(10, 3.3), gridspec_kw={'width_ratios': [1, 1, .8, .8, .8, .8]})
    stages = [('原始 RGB', pipeline['raw'], '210 × 160 × 3'),
              ('末两帧最大池化', pipeline['pooled'], '210 × 160 × 3')]
    stages += [(f'灰度帧 t−{3-i}' if i < 3 else '灰度帧 t', pipeline['stack'][i], '84 × 84') for i in range(4)]
    for ax, (title, frame, size) in zip(axes, stages):
        ax.imshow(frame, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        ax.set_title(f'{title}\n{size}', fontproperties=PANEL_FONT)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle('共享环境工厂输出的真实观测', fontproperties=TITLE_FONT)
    fig.tight_layout(rect=(0, 0, 1, .9), w_pad=.5)
    save(fig, 'observation_pipeline.png')

    fig, axes = plt.subplots(2, 4, figsize=(10, 4.8))
    event_data = {}
    for row, (label, sequence, color) in enumerate([
        ('成功接球', successes[0], '#267f70'), ('失球', failures[0], '#a6523b')]):
        event_data[label] = []
        for col, item in enumerate(sequence):
            ax = axes[row, col]
            ax.imshow(item['image'][135:210], extent=(0, 160, 210, 135), interpolation='nearest')
            if row == 0 or col < 3:
                ax.scatter(*item['ball'], s=110, facecolors='none', edgecolors='#ffd46b', linewidths=1)
            short_label = '接球' if row == 0 else '失球'
            ax.set_title(f'{short_label} {col+1}/4 · {item["frame"]/60:.2f}s\n'
                         f'{item["score"]}分 · {item["lives"]}命 · 帧{item["frame"]}',
                         fontproperties=PANEL_FONT, color=color)
            ax.set_xticks([])
            ax.set_yticks([])
            if col > 0:
                delta = item['paddle_x'] - sequence[col-1]['paddle_x']
                if abs(delta) > 1:
                    ax.annotate('', xy=(item['paddle_x'], 201), xytext=(sequence[col-1]['paddle_x'], 201),
                                arrowprops=dict(arrowstyle='->', color='#7cc9fa', lw=1.6))
            event_data[label].append({k: v for k, v in item.items() if k != 'image'})
            Image.fromarray(item['image']).save(DATA / f'behavior_{row}_{col}.png')
    fig.suptitle('同一模型、同一局游戏的接球与失球片段', fontproperties=TITLE_FONT)
    fig.tight_layout(rect=(0, 0, 1, .92), h_pad=1.3)
    save(fig, 'behavior_sequence.png')
    event_data['model'] = str(MODEL.relative_to(ROOT))
    event_data['model_sha256'] = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    event_data['seed'] = 42000
    event_data['pipeline_step'] = pipeline['step']
    event_data['pipeline_frame'] = pipeline['frame']
    (DATA / 'rollout_metadata.json').write_text(json.dumps(event_data, ensure_ascii=False, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=['training', 'distribution', 'diagnostics', 'robustness', 'rollout', 'all'], default='all')
    args = parser.parse_args()
    configure()
    for name, fn in [('training', training_curves), ('distribution', distribution),
                     ('diagnostics', diagnostics), ('robustness', robustness), ('rollout', rollout_figures)]:
        if args.only in (name, 'all'):
            fn()


if __name__ == '__main__':
    main()
