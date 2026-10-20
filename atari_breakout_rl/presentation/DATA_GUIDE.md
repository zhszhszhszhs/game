# PPT 实验数据说明

核对日期：2026-10-20。本文路径均相对于项目目录 `atari_breakout_rl/`。

## 1. 数据是否齐全

当前仓库已跟踪的 `report/data/`、`report/latex/images/`、配置和源码，加上本次 `.gitignore` 放行的数据，覆盖**现有实验结果的静态 PPT 制作**：任务与预处理、训练曲线、探索率与损失、最佳模型选取、逐局得分、低分风险、行为截图和实验参数。

此前已放行 45 个原始数据文件。本次补充四个核心实验的 `best_model/selection.json` 和 `results/environment.json`，总计 50 个文件，约 430 KiB；报告素材与已有 PPT 已在 Git 中，不计入这 50 个文件。

这表示文件可以正常加入 Git，修改 `.gitignore` 本身不会自动提交或上传数据。当前没有新增训练或评估。

**完整实验仍有缺口：最终留出测试、与推荐模型使用相同逐局种子的随机策略对照、多次独立训练结果尚未齐备。** 本文说明已有证据，不能把不存在的结果补成实验结论。

## 2. PPT 内容与数据对应

现有 PPT 为 [视觉增强定稿](output/Breakout_DQN_精修扩充版_v4_视觉增强定稿.pptx)，页码按该文件的 21 页计算；新增结果页可使用下面的完整数据。

| 内容 / 现有页码 | 原始依据 | 可直接使用的素材 |
| --- | --- | --- |
| 环境规则、动作与交互（3–5） | `src/fixed_breakout.py`、`src/env.py`；各训练 `config.json` 的 `environment` | `report/latex/images/game_live.png`、`report/data/dashboard_state.json` |
| 四帧输入与预处理（6） | `report/data/observation_samples.npz`、`src/env.py` | `report/latex/images/observation_pipeline.png` |
| DQN、回放、目标网络（7–10、19） | `TECHNICAL_REPORT.md`、`src/train.py` | 现有 PPT 中的教学示意 |
| 探索计划与训练参数（11–12、20） | `configs/dqn_custom.yaml`；实际实验使用 `logs/<run>/config.json` | `report/data/exploration_*.csv` |
| 四阶段训练、继续训练的波动（16，可扩充结果页） | `logs/<run>/train.monitor.csv`、`eval/evaluations.npz`、`training_summary.json` | `report/latex/images/training_curves.png`、`report/data/training_sources.json` |
| 学习率、探索率、损失诊断（可新增） | `logs/<run>/progress.csv` | `report/latex/images/training_diagnostics.png`、`report/data/loss_*.csv`、`diagnostic_sources.json` |
| 候选模型、30 局开发复测（15、21） | `results/evaluation/report_best_30/`、`report_final_30/` | `report/data/paired_scores.csv`、`paired_summary.json`、`report/latex/images/score_distribution.png` |
| 低分风险与配对差（可新增） | 同一组 30 局 CSV | `report/data/robustness_statistics.csv`、`robustness_summary.json`、`report/latex/images/score_robustness.png` |
| 模型行为、连续接球画面（可新增） | `report/data/behavior_*.png`、`rollout_metadata.json` | `report/latex/images/behavior_sequence.png` |
| 随机策略与选模过程（13、17，可补充已有记录） | `results/evaluation/custom_fixed_angle_random_development/`、`custom_fixed_angle_selection_20261004/` | 各目录 CSV / JSON；注意下文的协议差异 |
| 最佳节点选取（16、21） | `models/<run>/best_model/selection.json` | 最佳步数、均分、中位数、标准差、局数、评估种子 |

`report/data/manifest.json` 保存报告使用的模型 SHA-256、图像 SHA-256 和汇总信息，可用于识别素材版本。源文件中保留的 `/data/zhs/game/...` 是原运行机器的绝对路径，使用时按项目根目录定位对应相对路径。

## 3. 四个核心训练实验

`<run>` 仅指以下四个目录，位于 `logs/` 和 `models/` 下：

| run | 实验含义 | 最佳阶段步数 | 最佳均分 | 结束时均分 | 每次评估局数 |
| --- | --- | ---: | ---: | ---: | ---: |
| `dqn_custom_fixed_angle_500k` | 从头训练 | 350,000 | 174.00 | 133.30 | 20 |
| `dqn_custom_fixed_angle_refine_200k` | 一步低学习率续训 | 25,000 | 283.48 | 113.12 | 50 |
| `dqn_custom_fixed_angle_recovery_100k` | 三步回报恢复 | 0（起点） | 283.48 | 187.78 | 50 |
| `dqn_custom_fixed_angle_retention_100k` | 一步扩大回放 | 0（起点） | 283.48 | 216.94 | 50 |

恢复和扩大回放均在 50,000 步早停；目录名中的 `100k` 是预算，不是实际完成步数。四阶段分别计步；续训从基础训练的 350k 节点出发，两个恢复实验从续训最佳节点分支，不能简单拼成一条连续训练曲线。

每个训练目录放行六个文件：

| 文件 | 字段与用途 |
| --- | --- |
| `config.json` | 实际参数 `arguments`、环境契约 `environment`、依赖版本 `versions`、续训来源等。历史实验参数以它为准。 |
| `training_summary.json` | 完成状态、停止原因、实际步数、更新次数、耗时、回放容量等。 |
| `train.monitor.csv` | 首行为 `#` 开头的元数据，使用 `comment='#'` 读取；`r` 为回合奖励，`l` 为回合长度，`t` 为耗时，`lives` 为生命记录。按 `l` 累加形成训练步数横轴。 |
| `progress.csv` | 稀疏日志；主要列为 `time/total_timesteps`、`rollout/exploration_rate`、`train/learning_rate`、`train/loss` 和 `eval/*`。缺失损失不是零，不应补零。 |
| `eval/eval.monitor.csv` | 评估过程的回合奖励与长度记录。 |
| `eval/evaluations.npz` | `timesteps` 为评估节点，`results` 为各节点的逐局回报，`ep_lengths` 为逐局长度；用 `results.mean(axis=1)` 重建均分曲线。 |

`models/<run>/best_model/selection.json` 补充正式选模依据。续训最佳节点的 50 局中位数为 **283.5**；现有 PPT 第 21 页写“未列出”，制作新版时可据此补充。

训练回调评估从 `eval_seed=31000` 初始化环境后运行连续游戏序列，不应描述为逐局强制重置到 `31000–31049`。不同节点评估局数也可能不同。

当前 `configs/dqn_custom_refine.yaml` 已启用早停，但历史续训 200k 的 `config.json` 中 `early_stop_patience=0`；不要用当前默认配置替代历史实验实际设置。

## 4. 独立逐局评估与统计口径

这里的“独立逐局”指按各局种子重置环境，仍属于开发评估，不是最终留出测试。

| 数据目录（均在 `results/evaluation/`） | 局数与协议 | 用途 |
| --- | --- | --- |
| `custom_fixed_angle_random_development/` | 随机策略 20 局，起始种子 30000 | 同物理版本的初步基线；均分 0.60、中位数 0 |
| `custom_fixed_angle_selection_20261004/` | 各候选 50 局，起始种子 40000 | 比较基础训练的 300k、350k、500k 节点；选中 350k，均分 237.68 |
| `report_best_30/` | 推荐最佳模型 30 局，种子 42000–42029 | 最新报告的逐局得分依据 |
| `report_final_30/` | 续训最终模型，同一组 30 个种子 | 与最佳模型配对比较 |
| `regression_audit/` | 较早的同组 30 局诊断，以及额外 Q 值诊断 | 交叉核对；不能与报告复测叠加成 60 个独立样本 |

每组 `dqn_results.csv` / `random_results.csv` 的主要字段为 `episode`（局号）、`seed`（种子）、`reward`（得分）、`length`（决策步数）、`truncated`（是否截断）、`remaining_lives`（剩余生命）。配套 JSON 保存汇总、环境、种子与软件版本；DQN 文件还记录模型路径。

两份报告复测可直接制作结果表：

| 指标 | 续训最佳模型 | 续训最终模型 |
| --- | ---: | ---: |
| 局数 | 30 | 30 |
| 平均分 | 260.27 | 122.40 |
| 中位数 | 257.00 | 101.50 |
| 第 25 百分位 | 210.00 | 36.75 |
| 低于 50 分 | 1/30 | 9/30 |
| 观察到的通关局数 | 0 | 0 |

`report/data/paired_scores.csv` 按种子配对，`difference = best_reward - final_reward`。最佳模型胜出 28/30 局，平均配对差 137.87 分。新版报告使用 `robustness_summary.json` 的 Bootstrap 95% 区间 **[95.77, 177.43]**：按完整种子对重采样 20,000 次，随机种子 20261017。旧 `regression_audit/comparison.json` 的区间不同，做同一张图时不要混用两次统计结果。

随机基线的种子、局数与 30 局复测不同，不能用它计算同种子的配对提升；50 局训练回调的 283.48 和 30 局复测的 260.27 也不能合并成一个测试均分。

`regression_audit/q_diagnostics.json` 使用额外种子 43000–43007，每个模型 8 局；`reference_states.npz` 为相关参考状态。Q 值误差和动作分歧只能作为诊断，不能单独证明退化原因。PPT 第 8–9 页的 Q 值和学习目标是教学假设，不是这些诊断的实测输出；第 12 页流程示意曲线也不是训练成绩。

## 5. 图像素材与重绘

直接制作 PPT 可使用 `report/latex/images/` 下七张 PNG，无需模型权重。预处理样本保存在 `report/data/observation_samples.npz`；行为原始帧为 `behavior_*.png`，位置、速度、生命等上下文见 `rollout_metadata.json`；网页截屏时的状态见 `dashboard_state.json`。

从项目目录执行以下命令，可用现有数据分别重绘四类统计图，不会重新训练或评估模型：

```bash
python report/generate_figures.py --only training
python report/generate_figures.py --only diagnostics
python report/generate_figures.py --only distribution
python report/generate_figures.py --only robustness
```

需要项目依赖，以及脚本指定的 Noto Sans CJK 字体；详见 [报告复现说明](../report/README.md)。

**模型权重仍未放行。** 默认不传 `--only` 会同时重新运行模型生成预处理及行为画面，需要 `models/dqn_custom_fixed_angle_refine_200k/best_model/best_model.zip`。重新运行两模型评估还需要该实验的 `final_model.zip`；两份权重合计约 52 MiB。已有静态图像和逐局数据足以制作 PPT，但上传这些数据不等于交付可以重新推理、训练的完整模型包。

经验回放、历史 checkpoint、TensorBoard 事件、控制台日志和旧规则实验继续忽略。`results/environment.json` 是补充软件/硬件快照；具体实验的版本与环境以该实验的 `config.json` 或评估 JSON 为准。

## 6. 提交前检查

在仓库根目录执行：

```bash
git status --short --untracked-files=all
git ls-files --others --exclude-standard atari_breakout_rl/logs atari_breakout_rl/results atari_breakout_rl/models
git diff --check
```

新增数据应为：24 个训练日志文件、21 个评估文件、4 个最佳节点选取记录、1 个环境快照。`report/`、配置、源码和现有 PPT 原本已跟踪，可随仓库取得；本说明及数据文件仍需正常 `git add`、提交和推送。
