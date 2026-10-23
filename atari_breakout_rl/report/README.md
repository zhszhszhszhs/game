# DQN 与 PPO 实验报告

报告按“公共基础 → DQN 完整实验 → PPO 完整实验 → 统一比较 → 总结”组织。DQN、PPO 各自在独立章节内连续介绍原理、网络与更新循环、训练配置及实验分析。共用环境和评价口径只在公共基础章说明，冻结模型的同种子 50 局测试集中在比较章。

| 章节 | 源码 | 主要内容 |
| --- | --- | --- |
| 1. 项目背景与目标 | [introduction.tex](latex/sections/introduction.tex) | 研究目标、工作内容与报告安排 |
| 2. 游戏环境与公共实验基础 | [shared.tex](latex/sections/shared.tex) | 游戏规则、观测与奖励、共用系统、评价指标与数据用途 |
| 3. DQN 方法与实验 | [dqn.tex](latex/sections/dqn.tex) | 原理、实现、训练、30 局开发复测、退化诊断与行为分析 |
| 4. PPO 方法与实验 | [ppo.tex](latex/sections/ppo.tex) | 原理、实现、开发选模、多次配对续训、消融与 KL 诊断 |
| 5. 统一测试与方法比较 | [comparison.tex](latex/sections/comparison.tex) | 50 局统一测试、PPO 最终 GAE 消融、机制与工程取舍 |
| 6. 总结与局限 | [conclusion.tex](latex/sections/conclusion.tex) | 项目总结、结论范围与后续验证 |

现有 DQN/PPO 公式、十张图片、开发结果及负结果均保留；另补充三张 PPO 分析图：多次配对续训、改进措施消融、KL 控制诊断，共十三张图片。PPO 共同起点与两种 GAE 设置的结果仍在原统一测试表中；新 KL 阶段使用另一组开局，仅按自身对照分析，不替换原 DQN/PPO 主比较候选。此次补图读取已有实验日志及已完成的测试，不重新训练或评估模型。报告继续不添加文末文献列表及正文引用标记。

报告正文和图注使用概括描述，不列出具体种子编号、个别回合标识、采样帧号及日志定位细节；关键超参数、训练预算、样本规模和实验结果仍保留必要数值。新图用“重复一、二、三”标识训练重复；本说明、数据快照和复现脚本保留精确对应关系，便于独立核对。

插图不含底部说明小字。数据图统一使用 10 英寸画布宽度、Noto Sans CJK Bold 主标题 16 pt 和子图标题 11 pt；图像尺寸与行为帧信息移至子图顶部。网页截图使用仅作用于导出的样式隐藏页脚、底部说明和空历史面板，并统一主标题字体。

- [PDF 报告](latex/基于DQN与PPO的Breakout课程实验报告.pdf)
- [LaTeX 源码](latex/main.tex)
- [自动生成的测试数值](latex/comparison_metrics.tex)
- [新增分析数值](latex/ppo_analysis_metrics.tex)
- [图片目录](latex/images/)

## 数据来源

| 图片 | 来源与核对方式 |
| --- | --- |
| 网页展示 | Firefox 截取实际网页运行画面，截图时暂停；完整状态见 `data/dashboard_state.json`。 |
| 预处理 | 共享 `make_env` 的实际物理帧回调和网络输入。最新灰度图与堆帧输入逐像素核对；样本见 `data/observation_samples.npz`。 |
| 训练曲线 | `logs/dqn_custom_fixed_angle_{500k,refine_200k,recovery_100k,retention_100k}/` 中的 `train.monitor.csv` 和 `eval/evaluations.npz`。四阶段分别计步；两项恢复实验从续训最佳节点分支。节点数值见 `data/training_sources.json`。 |
| 探索率与损失 | 基础训练和续训的 `progress.csv`，不插补缺失损失记录；损失中位数按 20 个日志点平滑。导出数据与来源见 `data/exploration_*.csv`、`data/loss_*.csv` 和 `data/diagnostic_sources.json`。 |
| 得分分布 | 两个已有模型在相同开局下各运行 30 局，未更新参数。逐局 CSV、环境契约、软件版本和统计分别位于 `results/evaluation/report_best_30/` 与 `report_final_30/`。配对表见 `data/paired_scores.csv`。 |
| 低分风险与配对差 | 同一组真实逐局分数的经验累积分布和配对差；分位数采用线性插值，Bootstrap 对完整回合对反复重采样。统计与方法参数见 `data/robustness_statistics.csv` 和 `data/robustness_summary.json`。 |
| 行为连续帧 | 推荐最佳模型在同一局中的两个片段，按等时间间隔采样，只展示下方接球区域。完整原始帧见 `data/behavior_*.png`；位置、速度、分数、生命和模型 SHA-256 见 `data/rollout_metadata.json`。 |
| PPO 网络 | 根据 `src/ppo_tricks.py` 与 `src/balanced_ppo.py` 绘制独立策略/价值 NatureCNN、四动作概率和价值回归结构，参数与实际 GAE 配置一致。 |
| PPO 训练 | `logs/ppo_custom_fixed_angle_gae{,_control}_500k_20261022/` 的完整 `progress.csv` 与 `eval/evaluations.npz`；两组均新增约 50 万步，只有 GAE lambda 不同。绘图节点见 `data/ppo_training_sources.json`。 |
| PPO 多次配对续训 | `logs/ppo_custom_fixed_angle_kl_study_20261022_{adaptive,control}_s{48,49,50}/` 的完整开发 NPZ、逐局开发 CSV 和 `selected_model.json`。图中的重复一、二、三依次对应训练种子 48、49、50；每个节点 32 局，星号标最终选择、空心菱形标均分最佳。导出见 `data/ppo_analysis/development_curves.csv`、`checkpoint_selection.csv`。 |
| PPO 改进消融 | 从各轮真实测试 CSV 计算改动减同轮对照的逐局差；四项已有历史区间在核对原始均分差后沿用原统计，辅助奖励与 KL 区间新计算。各行测试开局不同，不作跨轮排名；双学习率组合含早停预算差异。导出见 `data/ppo_analysis/ablation_effects.csv`、`ablation_pairs.csv`。 |
| KL 控制诊断 | 六次训练的 `progress.csv` 中策略学习率倍率、实际学习率、完整更新后采样 KL、EMA、概率比越界比例及 clip 范围；仅绘制有实际记录的行，不插补。指标按 CSV 输出步数定位，可能滞后于其对应更新；越界比例不等于实际被裁剪的梯度贡献。导出见 `data/ppo_analysis/kl_diagnostics.csv`。 |
| DQN/PPO 比较 | DQN 与随机策略的 50 局结果见 `results/evaluation/report_dqn_ppo_50/dqn/`；PPO 起点及两组结果见 `results/evaluation/ppo_gae_20261022/`。所有模型使用相同的留出开局，逐局配对分数见 `data/dqn_ppo_paired_scores.csv`。 |

原 DQN 开发复测的两个模型分别为：

```text
models/dqn_custom_fixed_angle_refine_200k/best_model/best_model.zip
models/dqn_custom_fixed_angle_refine_200k/final_model.zip
```

复测最佳模型均分 260.27、中位数 257.00；续训最终模型均分 122.40、中位数 101.50。最佳模型胜出 28/30 局，平均配对差为 137.87 分，复现原有开发诊断汇总。所有图片均基于实际观测或日志绘制，未从汇总均分反推逐局分数。

补充统计：最佳模型低于 50 分为 1/30 局，续训最终模型为 9/30 局；第 25 百分位得分分别为 210.00 和 36.75。在两局局部退步的开局中，最佳模型分别少得 198、9 分。当前配对差均值的 Bootstrap 95% 区间为 [95.77,177.43] 分，只反映固定模型在开发样本上的条件变异。

以上 30 局仍是开发复测，与以下新增的 50 局测试分别报告。

## 统一测试与比较

在相同物理规则、原始奖励、完整五生命游戏、统一时间上限和确定性 CPU 推理下，每个固定模型使用相同开局运行 50 局。随机策略均匀采样动作。PPO 模型和检查点由验证/开发数据选定，本次报告整理不再训练或依据测试更换主候选。

| 固定模型 | 均分 | 标准差 | 中位数 | 低于 50 分 | 时间截断 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 随机策略 | 0.88 | 1.03 | 1.00 | 50/50 | 0/50 |
| DQN | 273.36 | 84.28 | 266.00 | 0/50 | 0/50 |
| PPO 共同起点 | 129.98 | 119.51 | 84.00 | 18/50 | 0/50 |
| PPO λ=0.98（主候选） | 196.54 | 138.91 | 166.50 | 9/50 | 5/50 |
| PPO λ=0.99（补充消融） | 203.42 | 126.34 | 191.50 | 6/50 | 3/50 |

主 PPO 按开发均分 196.00 高于 171.09 选为 λ=0.98，不因 λ=0.99 的测试样本均分较高而更换。DQN 减主 PPO 的配对均分差为 76.82，胜/平/负 31/0/19，95% bootstrap 区间 [28.02, 125.24]。λ=0.99 减 0.98 为 6.88，区间 [-41.08, 54.68]，不足以确认调高 lambda 的额外收益。

区间对完整回合对反复重采样得到。它只衡量固定模型的回合差异；DQN 与 PPO 的多阶段训练历史、累计预算及搜索不同，不能据此作同预算算法排名。所有候选本次均未完整通关。

[统计及模型/CSV SHA-256](data/ppo_report_summary.json)、[汇总表](data/dqn_ppo_metrics.csv)、[逐局配对表](data/dqn_ppo_paired_scores.csv)保留完整证据。`report/generate_ppo_figures.py` 校验种子覆盖、环境和分数后生成图片及 `latex/comparison_metrics.tex`，正文表格从数值宏读取，避免抄写不一致。算法说明的既有核对来源为 [PPO 原论文](https://arxiv.org/abs/1707.06347)及 [Stable-Baselines3 PPO 文档](https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/ppo.html)；项目实现差异以本地代码为准。

## 新增 PPO 分析图与统计口径

KL 配对实验现已全部完成：三次续训的改进/对照测试均分分别为 224.26/170.66、181.54/215.04、213.78/210.44；每个模型在相同 141000–141049 上各测试 50 局。训练从同一冻结起点开始，每次实际新增 503808 步；开发集合为 131000–131031，未依据测试更换检查点。

三次配对差为 +53.60、−33.50、+3.34，平均差 +7.81。对训练重复对和共同游戏开局分别配对重采样 20000 次、RNG 种子 20261022，95% 区间为 [−47.29, 64.17]。它只覆盖给定冻结起点下的三个续训重复，不能把每设置 150 局解释为 150 次独立训练。旧实验的区间仍仅衡量固定策略的回合差异；所有区间均为探索性描述。

新增脚本 [generate_ppo_analysis.py](generate_ppo_analysis.py)检查六次运行的共同设置、环境、开发/测试逐局种子、完整训练状态和选模记录，确认两维 KL 区间与后台最终汇总一致。[数据快照](data/ppo_analysis/)保存六份派生 CSV，[证据清单](data/ppo_analysis/analysis_summary.json)记录全部原始输入及派生 CSV 的 SHA-256。图表和文字分析中的新数值宏为 `latex/ppo_analysis_metrics.tex`，与原统一比较数值分别维护。

## 复现命令

在项目根目录，使用 README 指定的 Python 环境：

```bash
PY=/data/zhs/game/.venv/bin/python

"$PY" src/evaluate.py \
  --model models/dqn_custom_fixed_angle_refine_200k/best_model/best_model.zip \
  --episodes 30 --seed 42000 --device cpu --threads 1 \
  --output-dir results/evaluation/report_best_30
"$PY" src/evaluate.py \
  --model models/dqn_custom_fixed_angle_refine_200k/final_model.zip \
  --episodes 30 --seed 42000 --device cpu --threads 1 \
  --output-dir results/evaluation/report_final_30

MPLCONFIGDIR=/tmp/breakout-report-matplotlib "$PY" report/generate_figures.py
```

新增 50 局 DQN/随机基线及 PPO 图表的复现命令：

```bash
"$PY" src/evaluate.py \
  --model models/dqn_custom_fixed_angle_refine_200k/best_model/best_model.zip \
  --episodes 50 --seed 121000 --device cpu --threads 1 --compare-random \
  --output-dir results/evaluation/report_dqn_ppo_50/dqn

MPLCONFIGDIR=/tmp/breakout-report-matplotlib "$PY" report/generate_ppo_figures.py
```

PPO 的原始 50 局测试已由训练任务自动完成；精确模型路径、逐局 CSV 和散列见上述统计文件，可用同一评估入口及 `--episodes 50 --seed 121000 --device cpu --threads 1` 复测。完整训练与选模协议见项目唯一的 [PPO 记录](../PPO.md)。新增绘图脚本支持 `--only architecture|training|comparison`，其中 `comparison` 同时更新数值宏。

重建本次新增的三张分析图及数值宏：

```bash
# 校验原始日志和测试文件，导出数据快照并绘图
MPLCONFIGDIR=/tmp/breakout-report-matplotlib "$PY" report/generate_ppo_analysis.py

# 仅从已交付的数据快照重画，不需要模型权重或训练日志
MPLCONFIGDIR=/tmp/breakout-report-matplotlib "$PY" report/generate_ppo_analysis.py --from-snapshot
```

绘图脚本默认重建训练、分布、诊断、风险、预处理与行为图；可用 `--only training|distribution|diagnostics|robustness|rollout` 单独重建。中文绘图使用本机 Noto Sans CJK 字体。

网页截图需要 Firefox 和可访问本地端口的环境。先在一个终端运行服务，再在另一个终端截图；脚本会启动模型、等待得分、暂停截图，随后停止本次演示并退出浏览器：

```bash
"$PY" src/dashboard.py --port 8502
# 另一个终端：
/data/zhs/game/.venv/bin/python report/capture_dashboard.py
```

编译 PDF（需要 TeX Live 中文支持和 Fandol 字体）：

```bash
cd report/latex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
cp build/main.pdf 基于DQN与PPO的Breakout课程实验报告.pdf
```

本机系统 TeX 缺少 `ctexart`；本次把 Ubuntu 的 `texlive-lang-chinese_2019.20200218-1_all.deb` 解包到 `/tmp/breakout-report-tex`，未安装系统包。可在上述两次编译命令前加上以下环境变量使用临时依赖：

```bash
TEXINPUTS=/tmp/breakout-report-tex/usr/share/texlive/texmf-dist/tex//: \
OPENTYPEFONTS=/tmp/breakout-report-tex/usr/share/texlive/texmf-dist/fonts/opentype//: \
TEXMFVAR=/tmp/breakout-report-texvar \
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
```

临时依赖目录不属于报告交付内容；在完整 TeX Live 环境中直接编译即可。上传 Overleaf 时须同时包含 `main.tex`、`comparison_metrics.tex`、`ppo_analysis_metrics.tex`、完整 `sections/` 和 `images/`，无需训练代码或本机临时依赖。
