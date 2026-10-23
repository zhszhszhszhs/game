# 基于 DQN / PPO 的 Breakout 智能体

研究生强化学习课程项目：自制固定球速 Breakout + Gymnasium + Stable-Baselines3 DQN / PPO（CnnPolicy）。原 DQN 方法讲解见 [技术报告](TECHNICAL_REPORT.md)，修改与实验结果见 [修改汇总](CHANGE_SUMMARY.md)。

完整课程报告见 [DQN 与 PPO 实验报告](report/README.md)及 [PDF](report/latex/基于DQN与PPO的Breakout课程实验报告.pdf)，按公共基础、DQN、PPO、统一比较和总结组织。

## 安装

使用 Python 3.10/3.11，进入项目目录：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
```

CPU 环境将 PyTorch 下载源换成 `https://download.pytorch.org/whl/cpu`。当前服务器解释器为 `/data/zhs/game/.venv/bin/python`；本机依赖快照见 `requirements-lock.txt`。

## 游戏和输入

游戏采用五条生命、6×18 块砖、两面砖墙，每排奖励为 7/7/4/4/1/1，最高分 864。球速固定 120 px/s，挡板固定 16 px，反弹角度由接球位置决定，不随击球次数变化。发球位置和方向由种子生成，开局及丢球后自动发球；奖励来自击砖，无奖励塑形。几何、碰撞和反弹是自制规则重建，当前物理版本为 3。

共享预处理：RGB 210×160×3 → 重复动作 4 帧并最大池化 → SB3 WarpFrame 灰度缩放 → 堆叠 4 帧 → uint8 4×84×84。动作含义从环境读取。完整游戏结束才终止，108000 物理帧时间上限为截断。

## 检查、随机基线和训练

```bash
python src/env.py --raw
python src/env.py
python -m unittest discover -s tests -v
python src/random_agent.py --episodes 20 --seed 30000

# 短训练验证接口
python src/train.py --run-name dqn_custom_fixed_angle_smoke_new \
  --total-timesteps 10000 --buffer-size 5000 --learning-starts 1000 \
  --eval-freq 5000 --checkpoint-freq 5000 --n-eval-episodes 3

# 从头训练；每次实验使用新的 run-name
python src/launch_training.py --config configs/dqn_custom.yaml \
  --run-name dqn_custom_fixed_angle_new --device auto
python src/launch_training.py --run-name dqn_custom_fixed_angle_new --status
tail -f logs/dqn_custom_fixed_angle_new.console.log
tensorboard --logdir logs/ --host 127.0.0.1 --port 6006
```

参数可由命令行覆盖，详见 `python src/train.py --help`。`auto` 自动选择 CUDA 或 CPU。基础配置预算 500k；DQN 使用 CNN、经验回放、目标网络、ε-greedy、Huber loss 和 Adam。训练保存 checkpoint、均分最佳及中位数最佳模型。训练预算不保证收敛。

## PPO 训练与评估

PPO 的方法、配置、历轮结果、运行命令和改进计划统一见 [PPO 文档](PPO.md)，后续实验也更新这一份记录。

GAE 0.99/0.98 对照已完成，50 局测试均分为 203.42/196.54，配对差区间跨零。有界 KL 策略学习率反馈与固定计划的三种子配对续训也已完成：每次新增约 500k 步，各做 50 局新种子测试，三次平均得分为 206.53/198.71，配对差 +7.81、两维区间 [−47.29, 64.17]，尚未确认稳定优势。实现已通过 49 项回归测试；完整协议、结果与统计限制见 PPO 文档。

```bash
# 从冻结起点启动新的 PPO 实验；每次使用不同的 run-name
python src/launch_training.py --algo ppo --config configs/ppo_custom_gae.yaml \
  --resume models/ppo_custom_fixed_angle_gae_source_20261022/balanced_control.zip \
  --run-name ppo_custom_fixed_angle_gae_new --device auto

# 查询当前后台任务
python src/launch_training.py --run-name ppo_custom_fixed_angle_gae_500k_20261022 --status
python src/launch_training.py --run-name ppo_custom_fixed_angle_gae_control_500k_20261022 --status
```

参数、独立评估和产物说明见 [运行指南](PPO.md#usage)，已有结果与统计限制见 [历史实验](PPO.md#history)。

## DQN 最佳模型和续训

当前推荐模型：`models/dqn_custom_fixed_angle_refine_200k/best_model/best_model.zip`。50 局开发评估均分 283.48，额外相同的 30 个开发种子复测均分 260.27；两套评估分别报告。后续三步回报与扩大回放实验均未超过起点，已早停。冻结模型在 121000–121049 的 50 局统一测试均分为 273.36，完整结果与 PPO 比较见 [报告](report/README.md#统一测试与比较)。

```bash
python src/launch_training.py --config configs/dqn_custom_refine.yaml \
  --run-name dqn_custom_fixed_angle_refine_new \
  --resume models/dqn_custom_fixed_angle_refine_200k/best_model/best_model.zip
```

续训配置用原策略预热新 replay，开始前评估起点，连续 3 次未创新高自动早停；预热不消耗早停耐心。均分最佳模型同步保存压缩 replay。模型 ZIP 不包含 replay，有配套 replay 时可传 `--replay-buffer 路径`，其环境、容量及回报跨度必须匹配。`--n-steps` 保留多步回报实验能力。

`total-timesteps` 表示本阶段新增步数。50k 图像回放约占 2.8 GB 系统内存；压缩只减少磁盘占用。Ctrl+C/SIGTERM 保存中断产物。

## 网页展示

```bash
python src/dashboard.py --port 8501
# 在本地电脑执行
ssh -N -L 8501:127.0.0.1:8501 用户名@服务器地址
```

打开 **http://127.0.0.1:8501**，也可通过 VS Code 转发 8501。页面名称为「Breakout 游戏演示」，支持模型选择、随机策略、暂停、全屏和播放倍率。1× 对应目标 60 个物理帧/秒，CPU 推理。列表展示当前任务的正式模型；历史模型和短训练验证产物不进入列表。

## 评估、绘图和视频

```bash
python src/evaluate.py \
  --model models/dqn_custom_fixed_angle_refine_200k/best_model/best_model.zip \
  --episodes 50 --seed 10000 --compare-random --output-dir results/evaluation/custom_final
```

Random 和 DQN 使用相同环境及逐局种子。开发数据用于选模型，最终测试使用留出的种子。绘图与视频按需执行：

```bash
python src/plot_results.py --log-dir logs/dqn_custom_fixed_angle_refine_200k --window 100
python src/record_video.py \
  --model models/dqn_custom_fixed_angle_refine_200k/best_model/best_model.zip --name dqn_final
```

## 目录与排错

| 目录 | 用途 |
|---|---|
| `src/`、`src/web/` | 游戏、预处理、训练、评估、网页、绘图与视频入口 |
| `configs/` | DQN 从头训练、DQN 续训和 PPO 配置 |
| `tests/` | 游戏物理、模型接口、回放、回调和网页检查 |
| `models/<run>/` | 模型、checkpoint、replay；历史正式实验产物保留 |
| `logs/<run>/` | 配置快照、TensorBoard、进程状态、评估与训练总结 |
| `results/` | evaluation、figures、videos |

PPO 与 DQN 一样保留可提交 GitHub 的配置、训练/评估指标、逐局分数和选模记录；模型权重及运行日志保留本地。保留范围和核验命令见 [PPO 数据保留说明](PPO.md#实验数据保留与-github-提交)，文件清单见 [SHA-256 清单](report/data/ppo_artifacts.sha256)。

CUDA 不可用时在宿主环境检查 `nvidia-smi` 和 PyTorch；沙箱看不到 GPU 不代表 GPU 故障。内存不足时降低 buffer-size。没有 loss 时检查是否超过 learning-starts。原始 Gymnasium step 返回 5 项，SB3 VecEnv 返回 4 项；共享环境工厂处理跳帧、堆帧和终止。

当前代码只支持物理版本 3，旧 ALE 或旧规则模型会明确拒绝加载。历史实验参数在各日志的 `config.json` 中；训练与网页启动前先激活 Python 环境。
