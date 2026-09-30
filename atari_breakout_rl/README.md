# 基于 DQN 的 Atari Breakout 智能体

Deep Reinforcement Learning for Atari Breakout Using Deep Q-Network

研究生强化学习课程项目，使用 Gymnasium + ALE 运行游戏，Stable-Baselines3 DQN（CnnPolicy）学习控制挡板。提供训练、随机基线、模型评估和网页游戏演示。

## 安装

推荐 Python 3.11。在项目目录执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
```

CPU 环境将 PyTorch 下载源换成 `https://download.pytorch.org/whl/cpu`。现代 ale-py wheel 包含 ROM，无需安装旧版 Gym 或 AutoROM。本机已安装的环境可直接用 `/data/zhs/game/.venv/bin/python`；精确版本见 `requirements-lock.txt`。

## 环境与方法

- 状态：RGB 画面经过跳帧、最大池化、灰度化、缩放、4 帧堆叠，得到 `4×84×84` 图像。
- 动作：从环境读取 NOOP、FIRE、RIGHT、LEFT 的编号。
- 奖励：ALE 原生得分，不裁剪、不增加 shaping；所有生命耗尽才算完整游戏结束，保留时间上限。
- 统一预处理：ALE `frameskip=1`，SB3 AtariWrapper 跳帧 4 次，避免重复跳帧；开局和丢失生命后自动发球。
- 训练、评估和网页演示共用 `src/env.py`。关闭 sticky actions，结果不直接对比 ALE-v5 默认协议榜单。

CNN 提取画面特征，Online Q Network 预测各动作的折扣累计回报。Replay Buffer 随机抽取历史经验；Target Q Network 定期同步参数，提供较稳定的学习目标；ε-greedy 在随机探索和贪心动作之间选择。

```text
y = r + γ × (1 - terminated) × max_a Q_target(s_next, a)
```

教学中的平方 TD 误差为 `E[(y - Q(s,a))²]`；SB3 实际使用 Huber loss，通过 Adam 梯度更新网络。

## 运行

以下命令在项目目录执行，每次新训练使用不同的 `--run-name`。

### 环境检查与随机基线

```bash
python src/env.py --raw --seed 42
python src/env.py --seed 42
python -m unittest discover -s tests -v
python src/random_agent.py --episodes 20 --seed 10000
```

环境检查运行完整一局并打印动作含义与输入形状；预处理后的批次形状应为 `(1, 4, 84, 84)`。桌面环境可加 `--human`。

### 训练与进度

```bash
# 10k 步接口验证
python src/train.py --config configs/smoke.yaml --run-name smoke_new

# 后台启动新的 500k 步训练
python src/launch_training.py --config configs/dqn.yaml   --run-name dqn_new_500k --device auto

# 查询已经启动的修正训练
python src/launch_training.py --run-name dqn_revised_500k --status

# 训练指标
tensorboard --logdir logs/ --host 127.0.0.1 --port 6006
```

`auto` 自动选择 CUDA 或 CPU，也可指定 `--device cuda:0`。参数集中在 `configs/dqn.yaml`，命令行可覆盖，完整选项见 `python src/train.py --help`。

当前修正阶段 `dqn_revised_500k` 从旧 150k 最佳模型加载，计划额外训练 500k 步。已修复丢球后等待发球的问题；经验池增至 100k，先收集 50k 步经验，ε 在前 300k 步降至 0.05，Target Network 每 1k 步同步。修正后的 10k 短训练已通过，不能据此认定已经学好。

实时进度以 `logs/<run>/runtime.json` 和 `logs/<run>.console.log` 为准，不在文档中维护步数快照。每 50k 步保存 checkpoint，并进行 5 局开发评估以选择最佳模型。TensorBoard 包含 reward、episode length、epsilon、loss 和 learning rate；warm-up 阶段没有 loss 属正常现象。

续训示例（模型存在后使用）：

```bash
python src/launch_training.py --run-name dqn_continue   --resume models/dqn_revised_500k/final_model.zip   --total-timesteps 500000 --device auto
```

续训步数表示新增步数。默认重新计步并重启探索日程；可用 `--no-reset-timesteps` 保留累计计数。模型 ZIP 不含经验池，缺少 replay 时会重新收集经验；使用 `--replay-buffer 路径.pkl` 可加载匹配的经验池。

默认在结束或可捕获中断时保存 replay，100k 图像经验池约占 5.64 GB 内存及相近磁盘空间；周期 checkpoint 只保存模型。前台 Ctrl+C 或对后台训练 PID 发送 SIGTERM 会保存中断模型，强制终止无法保证保存。

### 网页游戏界面

服务器启动：

```bash
python src/dashboard.py --port 8501
```

本地电脑转发端口：

```bash
ssh -N -L 8501:127.0.0.1:8501 用户名@服务器地址
```

浏览器打开 **http://127.0.0.1:8501**。也可使用 VS Code 的端口转发。页面展示真实游戏画面，可选择随机策略或模型，开始、暂停、停止。演示使用独立 CPU 推理，不更新模型；要观看新 checkpoint，选择后重新开始。

### 后续评估、绘图与视频

以下脚本已提供，按当前要求暂不执行：

```bash
python src/evaluate.py --model models/dqn_revised_500k/best_model/best_model.zip   --episodes 50 --seed 10000 --compare-random
python src/plot_results.py --log-dir logs/dqn_revised_500k --window 100
python src/record_video.py --episodes 3 --name random_agent
python src/record_video.py --model models/dqn_revised_500k/best_model/best_model.zip   --episodes 3 --name dqn_final
```

独立评估使用贪心动作，与随机策略采用相同环境和 episode seeds，输出逐局得分、长度及统计对比。绘图生成 reward、平滑 reward、episode length 和 DQN/Random 对比图。旧实验未使用丢球自动发球规则，旧随机基线需要按新规则重测后再比较；训练探索得分也不能代替独立测试得分。

## 文件与输出

| 位置 | 内容 |
|---|---|
| `src/` | 环境、训练、回调、后台启动、评估、绘图、视频和网页代码 |
| `configs/dqn.yaml` / `configs/smoke.yaml` | 正式训练 / 短训练配置 |
| `tests/` | 环境接口、模型保存加载和网页测试 |
| `models/<run>/` | `checkpoints/`、`best_model/best_model.zip`、最终或中断模型、replay |
| `logs/<run>/` | 实验配置、Monitor CSV、TensorBoard、运行状态和训练总结 |
| `results/evaluation/` | 逐局 CSV、统计 JSON、Random/DQN 对比 |
| `results/figures/` / `results/videos/` | 报告图片 / 游戏视频 |

## 常见问题

| 问题 | 处理 |
|---|---|
| ALE environment / ROM not found | 确认 Python 环境和 ale-py 安装正确；使用 `gym.register_envs(ale_py)` 与 `ALE/Breakout-v5` |
| render 错误 | SSH 环境使用默认 RGB 渲染或网页，不加 `--human` |
| CUDA unavailable | 在同一运行环境检查 `nvidia-smi`、PyTorch CUDA 版本及设备权限；沙箱看不到 GPU 不代表宿主 GPU 故障 |
| Gym API / wrapper 不兼容 | 使用 Gymnasium；原始 step 返回 5 项，SB3 VecEnv 返回 4 项；共用 `make_env`，不要重复添加跳帧或堆帧 |
| 内存不足 | 减小 `--buffer-size`；经验池主要占系统内存 |
| 没有 loss / TensorBoard 为空 | 检查是否超过 learning_starts，以及日志目录是否正确 |

参考：[SB3 DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)、[ALE](https://ale.farama.org/getting-started/)、[SB3 Zoo Atari 配置](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml)。
