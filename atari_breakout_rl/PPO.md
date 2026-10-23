# PPO 方法与实验记录

PPO 的方法、历史结果、运行命令和后续计划统一维护在本文，后续实验直接更新对应章节，不再按轮次新增 Markdown 文件。实验日期为 2026-10-22；参数、版本、逐局分数及模型 SHA256 以各运行目录中的原始记录为准。

- [当前实验：KL 自适应学习率](#current)
- [方法与配置](#methods)
- [历史实验与结论](#history)
- [运行命令与产物](#usage)
- [验证及后续改进](#next)
- [参考资料](#references)

<a id="current"></a>
## 最新实验：KL 自适应策略学习率（已完成）

本轮研究有界的、仅作用于策略分支的 KL 学习率反馈。实现及配置为 [adaptive_kl_ppo.py](src/adaptive_kl_ppo.py) 和 [ppo_custom_adaptive_kl.yaml](configs/ppo_custom_adaptive_kl.yaml)。上一轮两组最终 rollout KL 的中位数分别约为 0.000207、0.000210，只有较小的策略变化；这提供了验证更新幅度的理由，但不能据此认定学习率不足或保证提高得分。

- **改进组**：用整轮更新后的采样 KL 做 EMA（alpha=0.2），目标 0.0005、上下阈值为目标的 1.5 倍及 1/1.5。KL 偏小时，下一轮策略学习率倍率乘 1.1；偏大时除以 1.1，限制在基础退火计划的 0.5–2.0 倍。瞬时 KL 过大时立即下调，不等待 EMA。原 KL 停更保护保留。
- **匹配对照**：同一实现关闭反馈，倍率固定为 1；策略/价值均按 2e-5→5e-6 退火。两组的价值学习率、Adam 分组、梯度裁剪、Huber、原始奖励、rollout、GAE 和评估协议相同；不启用额外价值拟合。
- **重复验证**：从上一轮按开发均分选中的 λ=0.98 最佳模型冻结副本开始，续训种子 48、49、50 各做一对，共 6 次训练，各计划新增 500000 步、完成 rollout 后实际 503808 步。它们是给定同一起点的续训重复，不是三个独立从零训练。
- **新数据划分**：开发种子 131000–131031、每次 32 局；测试种子 141000–141049、每模型 50 局。选模沿用开发均分与中位数的 5% 约束规则，训练执行完整预算，不按测试分数换主方案。原报告的 121000–121049 成绩不用于本轮选模。

实验名 `ppo_custom_fixed_angle_kl_study_20261022`，最多同时运行两项任务，其余自动排队。调度器会依次完成 6 次训练、各自自动测试，以及共同起点的同种子测试，最后生成 `comparison.csv` 和 `comparison_summary.json`。统计按训练种子对和共同游戏种子两个维度配对重采样；仅三个续训种子，区间仍是探索性证据。

正式队列已于 **2026-10-22 19:40（北京时间）**在 `cuda:0` 启动，调度 PID 为 **1204281**；首对 seed=48 的改进/对照 PID 为 **1204545 / 1204546**，均已确认使用 CUDA 并开始固定种子的起点评估。后续两对在槽位空出后自动启动。训练和测试不依赖当前对话保持打开。

**全部任务已于 23:28 完成**，调度状态为 `completed`，7/7 项任务成功结束，包含六次训练及各自测试和共同起点测试。每次训练实际新增 503808 步，固定测试种子为 141000–141049。

| 续训种子 | KL 反馈均分 | 固定计划均分 | 配对均分差 |
|---|---:|---:|---:|
| 48 | 224.26 | 170.66 | +53.60 |
| 49 | 181.54 | 215.04 | −33.50 |
| 50 | 213.78 | 210.44 | +3.34 |
| 三次平均 | 206.53 | 198.71 | +7.81 |

共同起点在同一组 50 局上的均分为 194.36。KL 减对照的训练/游戏两维配对 bootstrap 95% 区间为 **[−47.29, 64.17]**，三次中有提高也有退步，尚不能确认稳定优势。区间为给定冻结起点的三个续训重复所作探索性统计，不把每设置 150 局当成 150 次独立训练。最终数据见 [比较汇总](results/evaluation/ppo_custom_fixed_angle_kl_study_20261022/comparison_summary.json)。

LaTeX 报告已补充多次配对续训、历轮消融及 KL 控制诊断三类图；[分析快照](report/data/ppo_analysis/)保留绘图数据和源文件散列，复现方式见 [报告说明](report/README.md)。

[冻结协议](results/evaluation/ppo_custom_fixed_angle_kl_study_20261022/protocol.json)包含源模型和所有执行代码/配置的 SHA256。排队任务执行冻结代码，避免后续项目编辑改变同一轮对照。当前状态读取 `logs/ppo_custom_fixed_angle_kl_study_20261022/runtime.json`，各子任务状态见对应 `logs/<run>/runtime.json`；不在文档中手动维护步数。

方法参考 [skrl KL 学习率调节](https://skrl.readthedocs.io/en/develop/api/resources/schedulers/kl_adaptive.html)。本项目额外采用整轮采样 KL、EMA、策略分支独立倍率和原计划倍率上下限，属于待验证的实现变体，不等同于原论文的自适应 KL 惩罚目标。

## 上一轮结果：GAE 对照

本轮于 **2026-10-22 16:01（北京时间）**启动两组后台任务，两组均已完成新增 **503808 步**。训练种子均为 47，其余训练参数相同，仅比较 GAE lambda=0.99 与 0.98；训练后自动在 CPU 上各做 50 局独立测试。实时进度见 `logs/<run>/runtime.json`，不在本文重复维护不断变化的步数。

两组训练和独立测试均已完成，`post_evaluation.json` 状态均为 `completed`。相同 121000–121049 种子的结果如下：

| 模型 | 测试状态 | 50 局均分 | 中位数 |
|---|---|---:|---:|
| 共同起点 | 完成 | 129.98 | 84.0 |
| GAE 0.99 | 完成 | **203.42** | **191.5** |
| GAE 0.98 对照 | 完成 | **196.54** | **166.5** |

GAE 0.99 所选检查点为新增 503808 步，其测试样本均分较共同起点提高 73.44（约 56.5%），最高分 410。对照按开发规则选中 500000 步检查点，测试均分较起点提高 66.56。0.99 减 0.98 的配对均分差为 6.88，95% bootstrap 区间 [-41.08, 54.68]，胜/平/负为 25/0/25，尚不能确认提高 lambda 的额外收益。两组低于 50 分分别为 6/50、9/50，时间截断为 3/50、5/50，均未完整通关。

报告另对冻结的 DQN 推荐模型补跑相同 50 个种子，均分 273.36；随机策略 0.88。主 PPO 按开发均分 196.00 高于 171.09375 选择 λ=0.98，不按测试更换。DQN 与主 PPO 的配对均分差为 76.82，95% 区间 [28.02, 125.24]；两者训练历史及预算不同，只比较固定模型。上述区间均按完整种子对重采样 20000 次，随机数种子 20261022。详见 [LaTeX 报告](report/latex/main.tex)、[统一逐局分数](report/data/dqn_ppo_paired_scores.csv)及[统计与散列](report/data/ppo_report_summary.json)。

| 组别 | Run name | GPU | 启动 PID |
|---|---|---|---:|
| GAE 0.99 | `ppo_custom_fixed_angle_gae_500k_20261022` | cuda:1 | 333740 |
| GAE 0.98 对照 | `ppo_custom_fixed_angle_gae_control_500k_20261022` | cuda:3 | 333932 |

共同配置见 [ppo_custom_gae.yaml](configs/ppo_custom_gae.yaml)：16×256=4096 条 rollout、batch 256、4 个 epoch，学习率 2e-5→5e-6，clip 0.06→0.03，熵系数 0.0003→0.0001，gamma=0.995，Balanced PPO、原始奖励及新 Adam。本轮执行完整预算，不使用基于严格均分的提前停止，也不启用额外价值 epoch 或 3 倍价值学习率。

### 数据隔离与共同起点

源候选在验证前复制为不可变快照并计算 SHA256。源选择规则预先固定：保留均分达到候选最高均分 95% 的模型，再依次按中位数、25% 分位和均分选择。

| 源候选 | 新验证均分 | 中位数 |
|---|---:|---:|
| 原 Balanced 起点 | 93.6875 | 57.5 |
| DualRate 300k 稳定模型 | 135.78125 | 99.0 |
| Balanced 对照 200k 检查点 | **168.65625** | **103.5** |

选中的共同起点为 `models/ppo_custom_fixed_angle_gae_source_20261022/balanced_control.zip`。这是验证集选择结果，不是最终测试成绩。

| 用途 | 种子 | 局数 | 是否参与选择 |
|---|---|---:|---|
| 共同起点验证 | 101000–101031 | 每候选 32 | 仅选择源模型 |
| 训练开发评估 | 111000–111031 | 每次 32 | 选择检查点 |
| 最终独立测试 | 121000–121049 | 每模型 50 | 不参与模型或超参选择 |

两组起点在新开发集上的均分均为 **124.0625**、中位数 **94**。共同起点的最终测试基线已完成：50 局均分 **129.98**、中位数 **84**、最高分 **411**。三种用途的种子不同，不能把这些分数直接串成提升曲线。

本轮新增的评估与选模规则：

- **逐局固定种子**：8 个评估环境使用相邻初始 seed，各自后续每局 seed 增加 8，精确覆盖指定的 32 局。后续游戏不受前局 RNG 消耗影响；每轮保存按 seed 排序的 CSV。
- **带均分约束的稳定性选择**：保留均分最佳和中位数最佳两个模型；若后者均分达到前者的 95%，且中位数更高，则选后者做最终测试。5% 是预先约定的取舍阈值，不是显著性判据；原均分最佳权重仍保留。
- **诊断和心跳**：记录优势/returns 标准差、完整 rollout 上最终策略相对旧策略的近似 KL 和裁剪比例；评估每完成一局或约每 5 秒更新进度，长局未结束也能看到 transitions 增加。

gamma=0.995 时，GAE 权重的衰减尺度 `1/(1-gamma*lambda)` 从 lambda=0.98 的约 40 步增至 0.99 的约 67 步。这不是硬截断，更大的 lambda 也会增加方差，效果需由同条件对照确认。

本轮记录：[冻结协议与 SHA256](results/evaluation/ppo_gae_20261022/protocol.json)、[源模型选择](results/evaluation/ppo_gae_20261022/source_selection.json)、[起点测试](results/evaluation/ppo_gae_20261022/source_heldout/ppo_results.json)、[源验证分布图](results/figures/ppo_gae_20261022/source_validation.png)。开发协议已经改变，旧开发曲线不能直接拼接；本轮仍只有一个训练种子。

<a id="methods"></a>
## 方法与配置

标准 PPO 使用 SB3 2.9.0 `CnnPolicy`、GAE、裁剪策略目标、优势标准化、价值回归和熵奖励，每次只优化当前 rollout。与 DQN 共用 physics-v3 游戏：固定球速、五条生命、两面砖墙、原生最高分 864，完整游戏终止，108000 物理帧上限视为截断。输入为重复动作 4 帧并最大池化、灰度 84×84、堆叠 4 帧的 uint8 图像，策略内部除以 255。

| 配置 | 主要设置及用途 |
|---|---|
| [ppo_custom.yaml](configs/ppo_custom.yaml) | 标准 PPO：8×128 rollout、共享 CNN、学习率 2.5e-4、gamma 0.99、lambda 0.95、clip 0.1、熵系数 0.01 |
| [ppo_custom_refine.yaml](configs/ppo_custom_refine.yaml) | 独立策略/价值 CNN，8×512 rollout，学习率 5e-5→1e-5，clip 0.1→0.05，熵 0.003→0.001，价值权重 0.25、预测裁剪 1.0、KL 目标 0.01 |
| [ppo_custom_long_horizon.yaml](configs/ppo_custom_long_horizon.yaml) | gamma/lambda 0.995/0.98，学习率 1e-4→2e-5；辅助奖励组每接球 +0.1、每失球 −0.5，原奖励对照将两系数置零 |
| [ppo_custom_balanced.yaml](configs/ppo_custom_balanced.yaml) | 16×256 rollout，独立梯度裁剪、稳健价值损失，学习率 3e-5→1e-5，32 局批量开发评估 |
| [ppo_custom_dual_rate.yaml](configs/ppo_custom_dual_rate.yaml) | 价值学习率为策略的 3 倍，每轮额外 2 遍价值拟合；对照使用 Balanced、相同策略/价值学习率及无额外拟合 |
| [ppo_custom_gae.yaml](configs/ppo_custom_gae.yaml) | 已完成的 GAE 单因素实验及逐局固定种子、选模协议 |
| [ppo_custom_adaptive_kl.yaml](configs/ppo_custom_adaptive_kl.yaml) | 仅策略学习率的有界 KL 反馈，对照关闭反馈，三个续训种子配对验证 |

### 实现约定

- [训练入口](src/train_ppo.py)支持标准、Balanced 和 DualRate PPO。[模型加载器](src/model_io.py)按算法及变体加载模型，评估、视频和网页可使用相同接口。
- [网络拆分与熵退火](src/ppo_tricks.py)复制原策略和价值权重，迁移后初始预测不变。参数组改变时重置 Adam；匹配对照也重置 Adam。迁移后重设采样种子，避免网络构造消耗 RNG 改变初始采样状态。
- [BalancedPPO](src/balanced_ppo.py)要求两组参数无共享，分别以 0.5、1.0 裁剪策略与价值梯度；价值损失为 `2 * Huber(delta=10)`，小误差区与 MSE 相同，大误差的逐样本导数封顶为 20。SB3 更新的改写遵循 [MIT 许可证](src/SB3_LICENSE.txt)。
- [DualRatePPO](src/dual_rate_ppo.py)用两个 Adam 参数组保存独立学习率和状态。额外拟合只更新价值网络，策略梯度设为 `None`，同时冻结其 Adam 状态；保持原 rollout 的 returns、advantages、old values 和 old probabilities，不重算 PPO 概率比。
- [AdaptiveKLPPO](src/adaptive_kl_ppo.py)沿用独立 Adam 分组，但价值倍率固定 1、额外价值 epoch 固定 0。控制器只改变下一轮策略学习率，保存 EMA、倍率和更新计数；同变体完整续训保留这些状态，新阶段显式重置 Adam 时也重置控制器。关闭反馈的更新已与 Balanced 进行退化等价验证。
- [辅助奖励](src/training_reward.py)在 Monitor 外计算；原始得分始终独立记录，模型保存奖励系数，评估、视频和网页均使用原生得分。它改变训练目标，必须与原奖励对照区分。
- [逐局种子](src/episode_seeds.py)、[选模规则](src/checkpoint_selection.py)和[评估回调](src/callbacks.py)实现本轮评估协议。向量评估要求局数能被环境数整除；只有完整游戏结束才计为一局。

梯度范数或裁剪缩放不等于 Adam 参数步长按相同比例变化。Balanced 的 `train/value_loss` 是稳健损失，标准 PPO 对照是 MSE，绝对值不宜直接比较。更新后的价值误差表示对当前训练 targets 的拟合，不是独立泛化指标；最终比较以原始游戏得分为主。

<a id="history"></a>
## 历史实验与结论

以下每一轮内部使用相同测试种子比较模型；不同轮的测试种子和开发协议不同，均分不能直接横向解释为改进幅度。除特别说明外均为确定性动作，各轮只有一个训练种子；回合 bootstrap 区间只衡量固定策略在游戏种子上的差异，不代表跨训练种子的不确定性。

### 1. 标准 PPO

Run：`ppo_custom_fixed_angle_500k_20261022`，训练种子 42，实际 500736 步，RTX 4090 cuda:1，用时 20.47 分钟。开发集每次 10 局，最佳检查点为 400000 步、均分 58.0；曾在 450000 步降至 7.2，最终为 31.1，因此测试使用最佳检查点。

| 41000–41019，20 局 | 均分 | 标准差 | 中位数 | 最高分 |
|---|---:|---:|---:|---:|
| 随机策略 | 1.05 | 1.47 | 1.0 | 6 |
| 标准 PPO | 35.00 | 24.30 | 25.5 | 86 |
| 已有 DQN | 256.00 | 82.18 | 251.0 | 397 |

DQN 是已有多阶段精调模型，训练历史和预算不同，不能据此作同预算算法排名。原始产物：[训练总结](logs/ppo_custom_fixed_angle_500k_20261022/training_summary.json)、[模型 SHA256 与摘要](results/evaluation/ppo_custom_fixed_angle_500k_20261022/experiment_summary.json)、[逐局分数](results/evaluation/ppo_custom_fixed_angle_500k_20261022/ppo_results.csv)、[策略对比](results/evaluation/ppo_custom_fixed_angle_500k_20261022/comparison_all.csv)。

### 2. 稳定续训与单纯续训对照

两组从标准 PPO 400k 检查点出发，训练种子 43，各新增 200704 步，用时约 10 分钟。改进组采用独立 CNN、大 rollout、学习率/clip/熵退火等整组改动；对照延续原更新设置。开发最佳均分为改进组 71.7（新增 100k）、对照 60.0（新增 150k）。

| 模型 | 41000–41019 均分 / 中位数 | 51000–51029 均分 / 中位数 |
|---|---|---|
| 原 PPO | 35.00 / 25.5 | 34.00 / 27.0 |
| 单纯续训 | 52.45 / 31.5 | 53.00 / 33.5 |
| PPO + tricks | 56.15 / 43.5 | 43.33 / 26.0 |

新 30 局上，改进版相对原模型的配对均分差为 9.33，95% bootstrap 区间 [-10.03, 37.53]，胜/平/负为 13/2/15。组合额外收益不稳定，不能声称优于单纯续训。Run 名分别为 `ppo_custom_fixed_angle_refine_200k_20261022`、`ppo_custom_fixed_angle_control_200k_20261022`；[完整结果](results/evaluation/ppo_optimization_20261022/summary.json)。

### 3. 长回报与辅助奖励

两组从改进 PPO 最佳模型出发，训练种子 44，各新增 503808 步，约 55 分钟；gamma/lambda 改为 0.995/0.98，另比较接球/失球辅助奖励。20 局开发最佳均分：原奖励 103.30（400k）、辅助奖励 95.10（503808 步）。

| 61000–61049，50 局 | 均分 | 中位数 | 最高分 |
|---|---:|---:|---:|
| 该轮起点 | 50.34 | 29.5 | 306 |
| 长回报、原奖励 | 88.54 | 58.5 | 424 |
| 长回报、辅助奖励 | 57.34 | 48.0 | 256 |

后续沿用开发集选出的原奖励模型。当前单次实验未支持辅助奖励更优，也不能推断它普遍无效。Run 名为 `ppo_custom_fixed_angle_long_native_500k_20261022`、`ppo_custom_fixed_angle_long_shaped_500k_20261022`；[比较汇总](results/evaluation/ppo_long_horizon_20261022/comparison_summary.json)。

### 4. 独立梯度裁剪与 Huber

两组从长回报原奖励最佳模型出发，训练种子 45，各新增 503808 步。Balanced 最佳开发均分为 158.59375、中位数 80.5（503808 步）；标准 PPO 对照最佳均分 107.28125（400k）。本轮还把环境数改为 16、每环境 rollout 改为 256，总量仍为 4096，开发评估变为 8 环境、32 局。

| 81000–81049，50 局 | 均分 | 中位数 | 25% 分位 | 低于 50 分 |
|---|---:|---:|---:|---:|
| 该轮起点 | 89.28 | 62.5 | 35.25 | 42% |
| Balanced | 100.76 | 73.0 | 33.25 | 44% |
| 标准 PPO 对照 | 81.84 | 72.0 | 29.00 | 40% |

Balanced 相对起点的配对均分差为 11.48，95% 区间 [-29.02, 50.94]；相对对照为 18.92，区间 [-13.52, 52.16]。均分有所提高，低分尾部未改善，收益仍需重复验证；这也是两项改动的组合对照，尚未区分各部件贡献。

此前 1024 步、4 个 minibatch 的诊断中，策略/价值梯度范数中位数为 0.915/4.878，联合裁剪缩放中位数为 0.1266，仅支持梯度耦合的假设，不能直接推断 Adam 实际步长。Run 名为 `ppo_custom_fixed_angle_balanced_500k_20261022`、`ppo_custom_fixed_angle_balanced_control_500k_20261022`；[诊断](results/evaluation/ppo_balanced_20261022/gradient_diagnostic.json)、[逐局配对分数](results/evaluation/ppo_balanced_20261022/comparison.csv)、[统计汇总](results/evaluation/ppo_balanced_20261022/comparison_summary.json)。

### 5. 独立学习率与额外价值拟合

两组从 Balanced 最佳模型出发，训练种子 46。DualRate 的价值学习率为策略的 3 倍，每轮额外 2 遍价值拟合；对照使用相同策略/价值学习率、仅 4 遍共同更新。DualRate 在 400k 按严格均分规则早停，最佳均分仍来自起点；对照完成 503808 步。

| 91000–91049，50 局 | 均分 | 中位数 | 25% 分位 | 低于 50 分 |
|---|---:|---:|---:|---:|
| 该轮起点 | 128.12 | 101.0 | 41.75 | 32% |
| DualRate 按均分选中模型 | 128.12 | 101.0 | 41.75 | 32% |
| Balanced 对照 | 134.00 | 95.0 | 56.25 | 20% |

DualRate 测试选中的是起点，128.12 不是新增训练的收益。对照相对起点的配对均分差为 5.88，95% 区间 [-36.22, 48.74]；不足以认定改进稳定。

额外价值拟合在 97 条已记录更新中有 96.9% 降低训练 MSE，相对降幅中位数约 19%。更新前采样值的解释方差中位数约 0.647，额外拟合后对训练 targets 的解释方差约 0.911；二者不是同一种泛化指标，不能据此证明过拟合或策略改善。

值得复核的 300k 稳定模型：开发均分 158.53125，与起点 158.59375 仅差 0.0625，但中位数由 80.5 升至 132.5，25% 分位由 52.75 升至 54.75，低于 50 分比例由 25% 降至 21.875%。单检查点均分标准误约 23–25 分，不能可靠区分这样小的均分差；它已保存在 `stable_model`，并在本轮新验证集复核，见[当前实验](#current)。

Run 名为 `ppo_custom_fixed_angle_dual_rate_500k_20261022`、`ppo_custom_fixed_angle_dual_rate_control_500k_20261022`；[最终比较](results/evaluation/ppo_dual_rate_20261022/comparison_summary.json)、[诊断快照](results/evaluation/ppo_dual_rate_20261022/method_audit.json)、[冻结协议](results/evaluation/ppo_dual_rate_20261022/protocol.json)。诊断快照保留了对照尚未结束时的数据，以最终比较文件为准。

历史图表：[首轮回报](results/figures/ppo_custom_fixed_angle_500k_20261022/smoothed_reward.png)、[首轮开发曲线](results/figures/ppo_custom_fixed_angle_500k_20261022/development_evaluation.png)、[稳定续训对照](results/figures/ppo_optimization_20261022/comparison.png)、[梯度诊断](results/figures/ppo_balanced_20261022/current_diagnosis.png)、[Balanced 开发曲线与独立测试](results/figures/ppo_dual_rate_20261022/balanced_results.png)。

<a id="usage"></a>
## 运行命令与产物

以下命令在 `atari_breakout_rl` 目录、激活对应 Python 环境后执行。服务器已有 Conda 解释器为 `/data/zhs/game/.venv/bin/python`。每次启动使用新的 run name；预算表示新增交互步数，PPO 完成整个 rollout 才更新，实际步数可能向上取整。续训重新采样游戏，不复原游戏中间状态。

```bash
# 从零训练标准 PPO
python src/train_ppo.py --config configs/ppo_custom.yaml \
  --run-name ppo_custom_fixed_angle_new --device auto

# 从本轮冻结起点启动 GAE 实验；对照另用新 run name，加 --gae-lambda 0.98
python src/launch_training.py --algo ppo --config configs/ppo_custom_gae.yaml \
  --resume models/ppo_custom_fixed_angle_gae_source_20261022/balanced_control.zip \
  --run-name ppo_custom_fixed_angle_gae_new --device auto \
  --evaluation-dir results/evaluation/ppo_gae_recheck

# 查询当前两组后台任务
python src/launch_training.py --run-name ppo_custom_fixed_angle_gae_500k_20261022 --status
python src/launch_training.py --run-name ppo_custom_fixed_angle_gae_control_500k_20261022 --status

# 查询当前三种子 KL 实验的整体队列和各子任务
python src/launch_training.py --run-name ppo_custom_fixed_angle_kl_study_20261022 --status

# 创建并启动新的完整 KL 对照实验；两处 run-name 必须相同且从未使用
python src/run_ppo_study.py --prepare --run-name ppo_kl_new --device cuda:0 --workers 2
python src/launch_training.py --algo ppo-study --run-name ppo_kl_new \
  --protocol results/evaluation/ppo_kl_new/protocol.json

# 独立评估、随机对照、绘图和视频；模型路径也可换为 selected_model.json 中的路径
python src/evaluate.py \
  --model models/ppo_custom_fixed_angle_gae_source_20261022/balanced_control.zip \
  --episodes 50 --seed 121000 --compare-random --output-dir results/evaluation/ppo_recheck
python src/plot_results.py --log-dir logs/ppo_custom_fixed_angle_gae_500k_20261022 \
  --evaluation-dir results/evaluation/ppo_gae_20261022/ppo_custom_fixed_angle_gae_500k_20261022 \
  --output-dir results/figures/ppo_gae_recheck
python src/record_video.py \
  --model models/ppo_custom_fixed_angle_gae_source_20261022/balanced_control.zip --name ppo_demo
```

衰减配置续训需要 `--reset-timesteps`（各精调 YAML 已配置），确保学习率和 clip 衰减覆盖本阶段新增步数。`--reset-optimizer` 控制 Adam 重置，`--post-eval-episodes 0` 关闭自动测试。Ctrl+C/SIGTERM 会保存中断模型。

| 位置 | 内容 |
|---|---|
| `logs/<run>/config.json`、`training_summary.json` | 参数/版本/游戏规则、起点步数、预算、训练结果及设备 |
| `logs/<run>/runtime.json` | `training` 采样与更新，`evaluating` 开发评估，`testing` 独立测试；`completed`/`early_stopped` 为正常结束方式 |
| `logs/<run>/progress.csv`、Monitor、TensorBoard | 优化诊断与原始游戏回报；`optimization_epochs` 是 epoch 数，不是 minibatch 更新次数 |
| `logs/<run>/eval/evaluations.npz`、`episodes_<step>.csv` | 开发曲线与逐局种子分数；按种子配对使用 CSV，NPZ 的完成顺序不保证按 seed 排序 |
| `models/<run>/best_model/`、`stable_model/`、`checkpoints/`、`final_model.zip` | 均分最佳、中位数最佳、定期检查点和最终权重；`selection.json` 记录选择依据 |
| `logs/<run>/selected_model.json`、`post_evaluation.json` | 本轮最终选模与独立测试摘要；测试失败不会改写已完成的训练状态 |
| `results/evaluation/ppo_gae_20261022/<run>/ppo_results.{csv,json}` | 当前实验最终 50 局分数、模型路径及选模规则 |

网页模型列表、评估与视频支持 PPO 各变体。权重、图表、TensorBoard 等大文件保留本地；配置、指标、协议和测试结果按 `.gitignore` 约定纳入版本管理。

### 实验数据保留与 GitHub 提交

PPO 沿用 DQN 的关键数据保留方式：在原始目录中放行正式实验的配置、训练指标、评估分数和选模依据。本次范围为 2026-10-22 的基础训练、稳定续训及对照、长时域两组、Balanced 两组、Dual-rate 两组、GAE 两组，以及 KL 三个种子的六次配对续训，共 17 次训练。

| 位置 | 纳入 Git 的关键数据 |
|---|---|
| `logs/<run>/` | `config.json`、`training_summary.json`、`progress.csv`、所有训练 Monitor CSV；已有的 `selected_model.json`、`post_evaluation.json` |
| `logs/<run>/eval/` | `evaluations.npz`、单环境及向量环境的评估 Monitor CSV、已有的 `episodes_<step>.csv` |
| `models/<run>/{best_model,stable_model}/` | 两种选模方式各自的 `selection.json` |
| `results/evaluation/ppo_*/`、`report_dqn_ppo_50/` | 正式实验的逐局 CSV、汇总、比较、源模型选择和冻结协议 JSON，共 8 组评估目录 |
| `report/data/` | PPO 训练来源、DQN/PPO 配对分数及统计、`ppo_analysis/` 的派生 CSV 与源文件 SHA-256 |

保留范围由 `.gitignore` 中的正式实验目录白名单限定。模型权重与检查点 ZIP、replay、TensorBoard、控制台日志、`runtime.json`、锁文件、smoke 实验及冻结执行代码副本保留本地。报告插图和最终文档另由 `report/` 管理。

[数据校验清单](report/data/ppo_artifacts.sha256)覆盖上述 606 个数据文件，合计约 1.84 MiB；清单本身约 87 KiB，路径相对本项目目录。已有模型及冻结代码的 SHA-256 仍以各实验协议和报告统计为准。克隆仓库后可核验数据并从快照重画分析图；重新训练、模型推理和视频录制需要另外取得本地权重。

```bash
# 在 atari_breakout_rl/ 下核验本次数据快照
sha256sum --check --quiet report/data/ppo_artifacts.sha256

# 查看当前可新增提交的实验文件，遵循 .gitignore，无需 git add -f
git ls-files --others --exclude-standard -- logs/ models/ results/ report/data/
```

后续新增正式实验时，先补充目录白名单，再更新数据校验清单；保留逐局种子与开发/测试划分，避免只提交汇总均分。

<a id="next"></a>
## 验证及后续改进

本轮 KL 改动完成时 **49 项回归测试通过**。新增检查包括关闭反馈时的更新等价性、策略/价值学习率隔离、反馈上下限与尖峰响应、KL 停更保护、CNN 迁移预测不变、控制器与 Adam 状态保存后下一轮更新一致，以及配对统计输入验证。真实 Breakout 的短训练和自动测试用于验证运行链路，短训练分数不作为算法效果证据。既有测试还覆盖 SB3 退化等价性、Huber 导数、梯度预算隔离、额外价值拟合不修改策略及其 Adam 状态、固定种子与自动 reset、长局心跳和均分约束。

```bash
python -m unittest discover -s tests -v
```

后续按证据推进，每次隔离一个主要因素：

1. **完成三种子配对验证。** GAE 对照已经结束，未确认调高 lambda 的额外收益。本轮优先检验有界 KL 策略学习率反馈；报告每个训练种子的均分、中位数、25% 分位、低分比例、截断及配对区间，不只挑最好的一次。最终测试集不用于反复调参。
2. **探索和更新幅度继续分开消融。** 当前目标和倍率上限由旧训练诊断确定，不随本轮测试分数调整。若下一轮仍需要改善探索，再单独比较熵退火下限，避免把反馈控制和探索变化混为一项收益。
3. **表征与失败轨迹。** 先分析失球前 20–40 步及边缘反弹、第二墙、不同来球角度，再决定轻量残差 CNN、位置/速度辅助预测或多帧一致增广。模拟器坐标若作辅助标签需声明额外训练信息；推理仍读取像素。左右镜像需交换动作、处理非对称 HUD 和旧概率一致性。已有 4 帧速度信息，暂未证实需要 RNN/Transformer。
4. **价值尺度与消融。** Balanced 可作为候选基线，独立裁剪和 Huber 仍需拆开确认；不把 3 倍价值学习率与额外拟合设为默认。若新轨迹价值误差持续很大，再研究 PopArt 等目标归一化，并保存统计量、价值输出变换和 bootstrap 一致性。直接符号裁剪奖励会改变不同砖块分值的偏好。
5. **评估吞吐。** 已出现到达 27000 决策步上限的局，对照某阶段评估占用约 55% 总时间。心跳已落实；可进一步比较 CPU 批量与繁忙 GPU 吞吐、减少已完成槽位的无效推理、记录连续无得分时长。保留统一测试上限，并先判断长局是有效高分游玩还是无进展循环。
6. **后续独立方案。** 自动发球下 NOOP/FIRE 有重复语义，合并动作需概率映射或蒸馏及模型元数据迁移。若允许利用已有 DQN 教师，先在相同新种子上确认教师优势，再把蒸馏后 PPO 微调明确标为混合方法。

<a id="references"></a>
## 参考资料

[SB3 PPO 源码](https://stable-baselines3.readthedocs.io/en/v2.9.0/_modules/stable_baselines3/ppo/ppo.html)、[评估接口](https://stable-baselines3.readthedocs.io/en/v2.9.0/common/evaluation.html)、[PyTorch HuberLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.HuberLoss.html)、[优化器参数组](https://docs.pytorch.org/docs/2.14/optim.html)、[Spinning Up PPO](https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/ppo/ppo.html)。

[GAE](https://arxiv.org/abs/1506.02438)、[PopArt](https://arxiv.org/abs/1602.07714)、[Implementation Matters](https://arxiv.org/abs/2005.12729)、[Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)。这些资料支撑方法和实验设计，不构成本项目中特定改动已经有效的证据。
