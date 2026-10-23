# 强化学习课程实验报告

报告按六章组织：项目背景与目标、游戏环境与公共实验基础、DQN 方法与实验、PPO 方法与实验、统一测试与方法比较、总结与局限。两种方法各自连贯介绍原理、实现、训练及分析；DQN 的 30 局开发复测保留在 DQN 章，最终 50 局统一测试及 GAE 消融集中在比较章。数据来源与复现命令见 [上级说明](../README.md)。

`main.tex` 负责封面、排版和章节顺序，正文拆分在 `sections/` 中，统一测试数值由 `comparison_metrics.tex` 提供。原十张图片保留，PPO 章新增多次配对续训、改进消融和 KL 诊断三张图及文字分析，相关数值由 `ppo_analysis_metrics.tex` 提供。章节之间使用自动交叉引用。最终 PDF：[下载报告](基于DQN与PPO的Breakout课程实验报告.pdf)。

正文、表格文字和图注概述随机性控制与数据隔离方式，不列出具体种子编号、个别回合标识、采样帧号或模型文件的完整运行路径。新图以重复一、二、三标识三个训练重复；精确对应关系和来源散列保存在 [分析数据快照](../data/ppo_analysis/)及上级复现说明。关键超参数、预算、样本规模和结果保留必要数值。

## 编译

选择 XeLaTeX，连续编译两次以更新目录和交叉引用：

```sh
mkdir -p build
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
cp build/main.pdf 基于DQN与PPO的Breakout课程实验报告.pdf
```

需要带中文支持的 TeX Live 或 MiKTeX。源码使用 TeX 发行版自带的 Fandol 字体，不依赖本机商业字体。上传 Overleaf 时同时包含 `main.tex`、`comparison_metrics.tex`、`ppo_analysis_metrics.tex`、`sections/` 和 `images/`，并选择 XeLaTeX。直接编译不需要训练模型或 Python；新增分析图可以仅用已交付的数据快照重画。本机临时中文依赖的使用方式见上级说明。

## 源码组织

| 文件 | 职责 |
| --- | --- |
| `main.tex` | 封面、公共格式、数值宏与六章入口 |
| `sections/introduction.tex` | 项目背景、研究目标与阅读安排 |
| `sections/shared.tex` | 共用环境、预处理、系统接口和评价口径 |
| `sections/dqn.tex` | DQN 原理、实现、训练、开发结果与行为分析 |
| `sections/ppo.tex` | PPO 原理、实现、训练、开发选模与历轮结果 |
| `sections/comparison.tex` | 冻结模型 50 局测试、GAE 消融和方法比较 |
| `sections/conclusion.tex` | 项目总结、局限与后续验证 |
| `comparison_metrics.tex` | 绘图脚本生成的随机策略、DQN、PPO 测试数值 |
| `ppo_analysis_metrics.tex` | 三次 KL 配对续训及历轮消融的分析数值 |

六个大章节分别从新页开始，章节内部连续排版，不在小节前强制换页；每章结尾约束浮动图表，防止上一种方法的图表落入下一章。正文使用小四字号、1.25 倍行距；图表允许在当前位置、页顶或页底与正文共同排版，并统一收紧图文间距，减少章节内部的半空页。章节末尾为下一章独立起页保留自然留白。目录显示到二级标题，方法内部的原理、实现与训练细节使用三级标题。

## 封面

在源码开头修改 `MajorClass`、`StudentID`、`StudentName`、`Instructor` 等命令。封面已填写硕2605班、M202674292、赵恒升、何琨，课程名称为“强化学习”。

## 已补齐的图片

图片位于 `images/`，全部来自本地项目日志或实际运行：

| 文件 | 内容 |
| --- | --- |
| `game_live.png` | Firefox 截取推荐最佳模型的实际网页运行画面 |
| `observation_pipeline.png` | 环境回调采集 RGB、末两帧池化及实际四帧输入，最新灰度帧与网络输入逐像素一致 |
| `training_curves.png` | 四个正式实验的完整 Monitor 日志与开发评估 NPZ，标注最佳与早停节点 |
| `training_diagnostics.png` | 基础训练和续训的实际探索率与 Huber 损失，标记开发最佳节点 |
| `score_distribution.png` | 重新运行两个模型各 30 局后的真实配对分数与箱线图 |
| `score_robustness.png` | 同种子得分累积分布、低分风险与排序配对差 |
| `behavior_sequence.png` | 同一模型、同一局的接球与失球片段，标注时间、得分与生命 |
| `ppo_actor_critic.png` | 改进 PPO 独立策略/价值 CNN、四动作分布、价值目标及独立梯度裁剪 |
| `ppo_training_curves.png` | 相同起点与预算的 λ=0.98/0.99 开发均分、中位数、训练得分及整轮更新后 KL |
| `ppo_multiseed_curves.png` | KL 反馈与固定计划的三次配对续训，每次 32 局开发均分、中位数与所选检查点 |
| `ppo_ablation_effects.png` | 六项改动相对各自同轮对照的配对收益区间，保留组合改动与早停限制 |
| `ppo_kl_diagnostics.png` | 三次重复的学习率倍率、实际策略学习率、整轮 KL、EMA、越界比例及 clip 退火 |
| `dqn_ppo_comparison.png` | 相同 50 个测试种子的 DQN/PPO 分布和逐局配对差 |

可以在 `reportfigure` 调用中修改路径以使用 PDF、JPG 等格式。替换图片后应同步更新图注和分析文字；源码保留缺失图片时的占位回退逻辑。

新增 KL 分析采用另外一组开发与测试开局，不替换原统一比较。三个续训重复的平均配对收益为 +7.81，训练/游戏两维配对区间为 [−47.29, 64.17]，尚无稳定优势证据。历史行只使用固定策略的回合配对区间。图表、宏与快照由 `../generate_ppo_analysis.py` 统一生成，`--from-snapshot` 可检查快照散列并重画三张新增图。

文中区分原 DQN 的 30 局开发复测与最终 50 局测试。在统一留出的测试开局上，DQN、主 PPO λ=0.98、补充 λ=0.99 的均分分别为 273.36、196.54、203.42，共同起点为 129.98；主候选按开发数据选择，未按测试更换。[逐局配对分数](../data/dqn_ppo_paired_scores.csv)与[统计、环境及模型散列](../data/ppo_report_summary.json)可供复核。`comparison_metrics.tex` 由 `../generate_ppo_figures.py` 自动生成，更新数据时须同时重建图表和宏后重新编译。两种方法训练历史和预算不同，结论限于这些固定模型。
