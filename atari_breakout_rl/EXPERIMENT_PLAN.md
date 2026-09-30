# 实验设计与记录规范

主线仅使用 Random Agent 和 SB3 DQN，不加入 PPO 或其它算法。

1. 原始 ALE 环境 smoke test：至少一个完整 episode，验证 reset、step、RGB render。
2. Random baseline：先 20 episodes，种子 10000 起。
3. 预处理验证：uint8，单环境批次 `(1,4,84,84)`，frame skip 只执行一次。
4. 短训练：10,000 decisions，1,000 warm-up，验证梯度、replay、checkpoint、TensorBoard。
5. 正式训练：从头运行 500,000 decisions，seed=42；若后期 reward 持续明显上升，再按资源考虑延长。
6. 开发评估：训练内 EvalCallback，seed=1042 起，只用于 best model 选择。
7. 独立测试：seed=10000 起，DQN deterministic=True 与随机策略各 50 episodes，完全一致的环境配置。不得根据独立测试结果反复挑选 checkpoint。
8. 导出原始与平滑 reward、episode length、独立测试对比图；录制随机、早期与最终视频。

## 解释口径

- steps 是 agent decisions。每个动作最多重复 4 个 ALE frames，末步可能提前终止；reset 的随机 NOOP 与 FIRE 动作不计入决策步数。
- 所有策略统一 `repeat_action_probability=0.0`，与默认带 sticky actions 的 ALE-v5 基准不同，不能直接横向比较公开榜单。
- 不裁剪 reward，不用 reward shaping；不把丢失一条生命当作游戏结束。
- 自动 FIRE 只在 reset 时执行，丢失生命后是否再次 FIRE 由策略决定。
- 30 分钟仿真时间上限（108,000 ALE frames）避免不发球策略无限停留。
- Monitor 的 reward 是 reset 完成后的所有 step 原生 reward 之和；reset wrapper 的初始动作不计分。
- 训练 reward 包含 epsilon-greedy 探索；独立测试使用纯贪心动作，两种曲线/统计不能混为一谈。
- SD 是 episode 间的总体标准差（ddof=0），不是训练随机种子间标准差，也不是置信区间。
- 单次训练种子的课程项目不能支持“算法统计显著优于另一算法”等结论。
- 无论是否超越随机策略，都保留真实得分，不把短训练描述为已学会游戏。
