# 代码与训练验收记录

更新于 2026-09-30。按用户最新要求，不运行独立模型评估、绘图或视频录制。

| 项目 | 实测状态 |
|---|---|
| Python / PyTorch | 3.11.16 / 2.14.0+cu126 |
| Gymnasium / ALE / SB3 | 1.3.0 / 0.12.1 / 2.9.0 |
| pip check | 通过 |
| GPU | RTX 4090，CUDA forward/backward 通过；沙箱内需授权访问 |
| 原始 ALE | 完整 episode，reward=1，691 原始 steps，RGB 210×160×3 |
| Atari preprocessing | 完整 episode，reward=3，268 decisions，shape=(1,4,84,84) |
| 接口集成测试 | 2 项通过，包括实际推进4帧、frame stack、seed复现、模型训练保存加载 |
| 随机基线 | 20 局，mean=0.75，SD=0.7665，max=2，mean length=160.85 |
| 10k 短训练 | 已完成，GPU cuda:0，2250 updates，replay=5000，耗时120.81秒（含开发评估） |
| 短训练 checkpoint | 5k、10k、best_model、final_model 均已保存 |
| GPU 模型在 CPU 加载 | 已通过 |
| TensorBoard | reward/length/epsilon/loss/learning_rate 均存在且数值有限 |
| 500k 正式训练 | 进行中；完成状态见 logs/dqn_500k/training_summary.json |
| 独立评估 / 绘图 / 视频 | 仅编写脚本，按要求暂不运行 |

## 产物与日志

- `results/environment.json`：实际依赖与硬件。
- `results/smoke_raw.txt`、`results/smoke_preprocessed.txt`、`results/tests.txt`：环境验证。
- `results/evaluation/random_initial_20.csv` 和 `.json`：初始随机基线。
- `logs/smoke_10k/`：短训练的原始日志、配置、TensorBoard与训练总结。
- `models/smoke_10k/`：短训练模型。
- `logs/dqn_500k/`、`models/dqn_500k/`：正式训练。

## 结果解释

10k 模型的训练中单局贪心评估得分为0，达到时间上限；这是未充分训练策略可能不重新发球的真实结果，不能声称短训练已经学会接球。正式模型需要后续独立评估后才能给出泛化成绩。本次保留训练内 Evaluation Callback，仅用于最佳模型选择与训练监控。
