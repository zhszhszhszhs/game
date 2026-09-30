# 基于深度强化学习 DQN 的 Atari Breakout 游戏智能体

**Deep Reinforcement Learning for Atari Breakout Using Deep Q-Network**

研究生强化学习课程项目。使用 Gymnasium + Arcade Learning Environment (ALE) 的真实 Atari Breakout ROM，使用 Stable-Baselines3 (SB3) DQN，不自行开发游戏，也不重写 DQN。包含随机基线、训练、checkpoint、独立评估、TensorBoard、报告曲线和游戏视频。

## 本次交付范围与验证状态

按最新要求，本次只完成代码与训练；不执行独立模型评估、绘图或视频录制。对应脚本已编写，后续可按本文命令运行，其生成结果不作为本次已完成产物。随后按新增要求，另加实时网页游戏演示（不录制视频）。

已实测：原始 ALE 完整 episode、20 局随机基线、预处理/跳帧/堆帧、种子复现、模型保存加载、GPU 前向/反向、10k DQN 短训练，以及五类必需 TensorBoard 指标。10k 短训练完成 2,250 次梯度更新，replay 为 5,000 条；短训练内单局贪心开发评估为 0 分，不代表正式模型表现。

500k 正式训练的状态以 `logs/dqn_500k/training_summary.json` 为准；仅当 `status` 为 `completed` 才代表已结束。训练中可看 `logs/dqn_500k.console.log`。最新验收记录见 [TRAINING_STATUS.md](TRAINING_STATUS.md)。

## 1. 环境与任务

挡板反弹球，击碎砖块获得环境原生奖励。动作编号从 `env.unwrapped.get_action_meanings()` 读取，本机 Breakout 最小离散动作空间将在 smoke test 中打印。

| 强化学习元素 | 本项目定义 |
|---|---|
| 状态 | 连续 4 帧灰度图，uint8，`4×84×84` |
| 动作 | Breakout 最小离散动作空间（NOOP/FIRE/RIGHT/LEFT） |
| 奖励 | 每次 step 返回的 ALE 原生 reward，不裁剪、不 shaping |
| episode | 一局完整游戏（所有生命耗尽或达到 ALE 时间上限） |
| 策略 | 训练 epsilon-greedy；独立测试 `deterministic=True` |

所有流程调用同一个 `src/env.py:make_env`：

```text
ALE/Breakout-v5 (RGB 210×160×3, frameskip=1)
 → AtariWrapper (NOOP reset, frame skip 4, max pool, FIRE reset, grayscale, resize)
 → Monitor (原生得分，完整游戏)
 → DummyVecEnv (1 environment)
 → VecFrameStack (4 frames)
 → VecTransposeImage
 → uint8 (1, 4, 84, 84)
 → CnnPolicy (SB3 自动 /255 归一化)
```

只在 AtariWrapper 内跳帧一次。关闭 reward clipping、episodic life 和 sticky actions（`repeat_action_probability=0.0`）。`gym.register_envs(ale_py)` 显式注册现代 ALE。保留标准 FIRE reset；丢失生命后，重新发球由策略决定。ALE 每局最多 108,000 仿真帧，约 27,000 个决策步，防止不发球策略无限停留。Reset 中的 NOOP/FIRE 动作不计入 Monitor 统计。

本配置并非 ALE-v5 默认的 0.25 sticky-action 协议，分数不能直接与该协议的公开榜单比较。更多实验口径见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。

## 2. 目录

```text
atari_breakout_rl/
  README.md, requirements.txt, requirements-lock.txt
  EXPERIMENT_PLAN.md
  configs/dqn.yaml          # 正式训练参数
  configs/smoke.yaml        # 10k 短训练
  src/env.py               # 原始/预处理 smoke test、统一环境
  src/train.py             # DQN、日志、checkpoint、续训
  src/evaluate.py          # 独立评估与随机对比
  src/random_agent.py      # Random baseline
  src/record_video.py       # 独立加载模型并录制 MP4
  src/plot_results.py       # 四类报告图
  src/utils.py             # seed/device/CSV/JSON
  src/dashboard.py         # localhost 网页服务
  src/game_viewer.py       # 独立 CPU 游戏演示
  src/web/index.html       # 游戏画面与控制按钮
  tests/test_contract.py    # 跳帧、堆帧、seed、模型保存加载
  models/<run>/            # final_model.zip / checkpoints / best_model
  logs/<run>/              # Monitor CSV、progress.csv、TensorBoard、配置
  results/evaluation/      # 每 episode CSV、统计 JSON、comparison.csv
  results/figures/         # 200 dpi PNG
  results/videos/          # MP4、截图、视频 metadata
```

脚本默认输出路径相对于工程目录，不依赖当前 shell 的工作目录；自定义相对路径按 shell 工作目录解释。每次训练使用唯一 run-name，已有目录会报错，避免覆盖实验。

## 3. 安装

推荐 **Python 3.11**。本次版本与硬件信息记录在 `results/environment.json`；`requirements.txt` 使用经过验证的主要版本范围，`requirements-lock.txt` 记录本机精确安装版本（含 Linux CUDA 依赖，不要求其它平台照搬）。

在项目目录执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 本机 NVIDIA 535 驱动 + RTX 4090 已验证 CUDA 12.6 wheel 的实际运算。
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
```

CPU 机器把 PyTorch 安装命令换成：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

也可以用 Conda 创建环境：`conda create -n breakout python=3.11 pip`，然后 `conda activate breakout`。在当前交付机器，环境已经安装在 `/data/zhs/game/.venv`，可直接：

```bash
conda activate /data/zhs/game/.venv
cd /data/zhs/game/atari_breakout_rl
```

若不想激活，以下所有 `python` 可替换为 `/data/zhs/game/.venv/bin/python`。

现代 `ale-py` wheel 已包含 ROM，无需 AutoROM 或旧的 `accept-rom-license` extra。OpenCV 仅用于官方 AtariWrapper 灰度化与缩放，选用 headless 包。`imageio-ffmpeg` wheel 提供视频编码程序，不需要训练时打开显示窗口。

## 4. 按阶段运行

### 环境检查与随机基线

```bash
python src/env.py --raw --seed 42
python src/random_agent.py --episodes 20 --seed 10000 \
  --output results/evaluation/random_initial_20.csv
python src/env.py --seed 42
python -m unittest discover -s tests -v
```

两种 smoke test 均运行完整 episode，打印 observation/action space、shape、action meanings，并验证 RGB render。预处理后批次 shape 应为 `(1,4,84,84)`。桌面可加 `--human`；SSH/无显示器使用默认 RGB 模式。

### DQN 短训练

```bash
python src/train.py --config configs/smoke.yaml --run-name smoke_10k
```

短训练只有接口验收意义，不能证明已经学会游戏。`learning_starts` 必须小于 total steps，否则根本不会更新网络；程序会检查。

### 正式训练

```bash
python src/train.py --config configs/dqn.yaml \
  --total-timesteps 500000 --seed 42 --device auto --run-name dqn_500k
```

`auto` 在 `torch.cuda.is_available()` 为真时选择 CUDA，否则选择 CPU；显式 `--device cuda:0` 在 CUDA 不可用时直接报错。选卡例子：`CUDA_VISIBLE_DEVICES=2 python src/train.py ... --device cuda:0`。SB3 的普通 DQN 使用一张 GPU，本项目不需要多卡分布式训练。

所有核心参数均可 YAML 配置并被 CLI 覆盖：

```bash
python src/train.py --help
python src/train.py --total-timesteps 1000000 --seed 42 \
  --learning-rate 0.0001 --buffer-size 50000 --learning-starts 20000 \
  --batch-size 32 --gamma 0.99 --train-freq 4 --gradient-steps 1 \
  --target-update-interval 10000 --exploration-fraction 0.2 \
  --exploration-final-eps 0.05 --checkpoint-freq 100000 \
  --eval-freq 50000 --n-eval-episodes 5 --device auto \
  --log-dir logs --model-dir models --run-name dqn_1m
```

steps 是 agent decisions，不是 ALE 原始帧；500k steps 约对应 2M 仿真帧。默认 replay 50k 存储当前与后续堆叠帧，单图像部分约 2.82 GB 系统内存；减小 `--buffer-size` 可节省 RAM。默认不保存 replay 到磁盘，避免每个 checkpoint 再占数 GB。

每 100k steps 保存 `models/<run>/checkpoints/dqn_breakout_<steps>_steps.zip`；每 50k 使用独立开发环境评估 5 episodes，最佳模型保存在 `models/<run>/best_model/best_model.zip`，最终模型为 `models/<run>/final_model.zip`。Ctrl+C 会保存 `interrupted_model.zip`。定期 checkpoint 不依赖正常训练结束。

建议按 500k → 1M → 2M → 5M 增加预算，先观察曲线与资源，不保证固定步数达到高分。

### 续训

```bash
python src/train.py --resume models/dqn_500k/final_model.zip \
  --total-timesteps 500000 --run-name dqn_continue_1m
```

续训的 steps 表示**额外**步数。ZIP 含网络、优化器与算法参数，但不含 replay；缺少 replay 时会重新 warm-up。需要保留 replay 时在原训练加 `--save-replay-buffer`，续训加 `--replay-buffer 路径.pkl`。续训会重新建立环境，epsilon 的 schedule 随新的总训练预算计算，因此不声称与未中断的一次训练逐步一致。跨段画图时本脚本只画指定目录的训练段，并保存/应用起始步数偏移。

### TensorBoard

```bash
tensorboard --logdir logs/ --host 127.0.0.1 --port 6006
```

打开 `http://127.0.0.1:6006`。远程机器可用 SSH 端口转发。主要指标：`rollout/ep_rew_mean`、`rollout/ep_len_mean`、`rollout/exploration_rate`、`train/loss`、`train/learning_rate`，以及 `eval/mean_reward` 和 `replay/size`。Warm-up 前没有 training loss 是正常现象；均值曲线为最近 episode 的统计，逐 episode 原始数据在 `train.monitor.csv`。

### 独立评估与对比

```bash
python src/evaluate.py --model models/dqn_500k/best_model/best_model.zip \
  --episodes 50 --seed 10000 --compare-random
```

独立测试使用与训练内开发评估不同的种子。两种策略使用同一环境、相同 50 个 episode seeds，DQN 不更新网络、纯贪心选动作。输出每局 reward/length，均值、总体标准差、中位数、最大值、最小值、平均长度，并生成 `dqn_results.csv`、`random_results.csv`、对应 JSON 和 `comparison.csv`。允许 `--episodes 100`；正式报告至少 30。比较不同 checkpoint 时为每个结果指定独立 `--output-dir`，避免覆盖。

### 绘图

```bash
python src/plot_results.py --log-dir logs/dqn_500k --window 100
```

输出 `reward_vs_steps.png`、`smoothed_reward.png`、`episode_length.png`、`dqn_vs_random.png`。前几局 rolling mean 使用已有数据；柱图误差棒是 episode 标准差，不是置信区间。图来自完整游戏 Monitor CSV，标题和单位清楚，可直接插入 PPT/报告。

### 视频与截图

```bash
python src/record_video.py --episodes 3 --name random_agent
python src/record_video.py --model models/dqn_500k/checkpoints/dqn_breakout_100000_steps.zip \
  --episodes 3 --name dqn_early
python src/record_video.py --model models/dqn_500k/best_model/best_model.zip \
  --episodes 3 --name dqn_final
```

在 `results/videos/` 得到 MP4、首帧 PNG、每局 reward/seed/length 的 JSON。默认视频 seed=20000 起，和定量评估分开。逐帧流式编码，不把整局视频存进内存。取每个决策步的 RGB 画面，以 15fps 播放（60Hz/4），不改变输入预处理。VecEnv 结束后会自动 reset，录制代码跳过下一局的 reset 画面；末步终止帧因自动 reset 不再录入。

## 网页游戏演示与 SSH 端口转发

这是 **Breakout 游戏画面**，并非训练曲线面板。页面显示 ALE 真实 RGB 画面，可选随机策略或已保存的 DQN checkpoint，支持开始/重开、暂停/继续、停止，显示得分、剩余生命、当前动作与游戏局数。模型列表会自动发现新 checkpoint；正在演示的模型不会被训练中的新模型自动替换，需要选择并重新开始。

在服务器启动（无需新增第三方依赖）：

```bash
python src/dashboard.py --port 8501
```

在**本地电脑**终端执行，服务器地址/用户名使用你原来的 SSH 连接信息：

```bash
ssh -N -L 8501:127.0.0.1:8501 用户名@服务器地址
```

保持该 SSH 窗口打开，在本地浏览器访问 **http://127.0.0.1:8501**。如果你已有 SSH 主机别名，也可以把 `用户名@服务器地址` 换成该别名。非默认 SSH 端口可添加 `-p SSH端口`。本地 8501 已占用时用 `-L 8502:127.0.0.1:8501`，浏览器打开 8502。

远程 VS Code 用户也可以在“端口 / Ports”面板转发服务器端口 8501。网页服务仅绑定服务器 loopback，不必开放服务器公网端口。

演示与训练进程分开：演示在 CPU 上推理，DQN 使用 `deterministic=True`；不更新权重、不录制文件、不替策略自动发球。原始 60Hz / frame skip 4，以 15 个决策每秒播放。如果早期模型停住不发球，属于模型真实行为，可重开或选择随机策略验证游戏画面。环境及预处理与训练保持一致。

游戏结束后自动开启下一局。关闭浏览器不会停止服务器端演示，离开前可点击“停止”。一个服务器实例共享一场演示，不同浏览器会看到同一局。

验证命令：`python -m unittest discover -s tests -p test_game_viewer.py -v`；服务启动后可用 `python tests/check_web.py --port 8501` 检查 HTTP 控制、随机策略/DQN 推理与真实 PNG 画面，检查完会停止演示。

## 5. DQN 原理（课程报告要点）

CNN 从堆叠图像提取球、挡板、砖块位置及运动信息，online Q network 输出每个动作的预期折扣累计回报 `Q(s,a)`。训练初期 Q 值不准确，epsilon-greedy 以概率 epsilon 随机探索，之后逐渐更多选择 Q 值最大的动作。

Replay buffer 保存 `(s,a,r,s',done)`，随机抽取 batch，减弱连续图像样本的相关性并复用经验。Target Q network 定期复制 online 网络参数，为 TD 目标提供相对稳定的参考。

一步 Bellman target（真正终止时不 bootstrap）：

```text
y_t = r_t + gamma * (1 - terminated_t) * max_a Q_target(s_{t+1}, a)
```

教学形式的平方 TD 误差为 `L = E[(y_t - Q_online(s_t,a_t))²]`。**SB3 2.9 实际使用 Huber loss** (`smooth_l1_loss`)，并进行梯度裁剪和 Adam 更新；不是自行实现 MSE。Gymnasium 的 truncation 与真正 termination 由 SB3 replay/VecEnv 按其接口处理。这里使用基础 DQN，未加入 Double/Dueling/Prioritized Replay 或 PPO。

奖励通过 Bellman 目标更新当前动作的 Q 值，未来获得奖励的行为沿时间传播回之前的状态，因此能逐渐学习有利于接球与得分的动作。训练不保证单调改善；高 episode length 也可能代表不发球停滞，需要结合 reward 与视频解释。

## 6. 常见问题

| 问题 | 检查与处理 |
|---|---|
| `ALE environment not found` | 确认运行的是安装依赖的 Python；`import ale_py; gym.register_envs(ale_py)`；使用 `ALE/Breakout-v5` |
| `ROM not found` | 检查 `python -m pip show ale-py`，重新安装 requirements 中的现代 wheel；不要混用旧 atari-py/AutoROM 教程 |
| render/display 错误 | 远程环境使用默认 `rgb_array`，不要加 `--human`；人类窗口需要可用桌面/SDL 显示服务 |
| CUDA unavailable | 同一终端执行 `nvidia-smi` 和 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`；检查容器/沙箱设备权限、CUDA wheel、驱动；暂时用 `--device cpu` |
| GPU 可用但占用不高 | Atari 模拟与单环境采样在 CPU，DQN 每 4 步才更新一次；GPU 不持续满载并不等于未使用 GPU |
| Gym API 不一致 | 不安装/导入旧 `gym`。原始环境 `reset→(obs,info)`、`step→5元组`；SB3 VecEnv `reset→obs`、`step→4元组`，done 自动 reset |
| wrapper/shape 不兼容 | 所有流程使用 `make_env`，不要额外套 AtariPreprocessing 或 FrameStackObservation；再次执行 smoke test 与 unittest |
| CUDA/模型加载错误 | 使用相同依赖与相同 observation/action space；加载时传入统一环境验证维度；可信本地模型才加载 |
| 内存不足 | replay 使用 RAM；降低 buffer size，保持大于 batch size。不要因 GPU 显存空闲就无限加大 replay |
| 安装临时目录无空间 | 选择有空间的数据盘目录，设置 `TMPDIR=/path/to/tmp python -m pip install --no-cache-dir ...` |
| 训练没 loss | 总步数必须超过 learning_starts；查看 training_summary 的 gradient_updates |
| 贪心策略不发球/长时间零分 | 保留真实 timeout 与低分；检查 FIRE 学习和视频，增加训练预算。不要只挑高分视频代替定量结果 |
| TensorBoard 为空 | 指向 `logs/`；确认 warm-up 后已训练，检查 `events.out.tfevents.*` 和 `progress.csv` |
| 视频编码失败 | 检查 imageio-ffmpeg wheel 与输出磁盘；不需要 OpenCV 的 GUI/video 编码功能 |

## 7. 报告组织与参考

建议顺序：背景 → 状态/动作/奖励 → Atari 预处理 → DQN/CNN/replay/target/epsilon → 实验配置 → 随机基线 → 训练曲线 → 独立测试 → 视频 → 局限与总结。使用真实 CSV，区分训练探索成绩与独立贪心测试成绩；单次训练不能支持多 seed 的显著性结论。

官方资料（开发时结合本机安装源码核对）：

- [ALE 安装与 ROM](https://ale.farama.org/getting-started/)
- [ALE 更新记录](https://ale.farama.org/release_notes/)
- [SB3 DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
- [SB3 AtariWrapper](https://stable-baselines3.readthedocs.io/en/v2.9.0/common/atari_wrappers.html)
- [Gymnasium API](https://gymnasium.farama.org/api/env/)
- [NVIDIA CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
