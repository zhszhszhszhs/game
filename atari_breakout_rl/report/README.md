# 报告插图与复现

2026-10-17 已运行项目，补齐 LaTeX 报告的七张图片并更新 PDF；Word 报告同步嵌入实验图表。图注和分析按实际数据修订。第四部分补充探索率与损失日志诊断，第五部分补充低分风险、分位数及配对差分析；已移除文末文献列表及正文引用标记。

插图已移除底部说明小字。六张数据图统一使用 10 英寸画布宽度、Noto Sans CJK Bold 主标题 16 pt 和子图标题 11 pt；图像尺寸与行为帧信息移至子图顶部。网页截图使用仅作用于导出的样式隐藏页脚、底部说明和空历史面板，并统一主标题字体。

- [PDF 报告](latex/基于DQN的Breakout课程实验报告.pdf)
- [Word 报告](基于DQN的Breakout智能体课程报告.docx)
- [LaTeX 源码](latex/main.tex)
- [图片目录](latex/images/)

## 数据来源

| 图片 | 来源与核对方式 |
| --- | --- |
| 网页展示 | `src/dashboard.py --port 8502`，Firefox 无界面浏览器截图；种子 42000，20 分、5 条生命。截图时暂停，完整状态见 `data/dashboard_state.json`。 |
| 预处理 | 共享 `make_env` 的实际物理帧回调和网络输入；种子 42000，第 30 决策步、第 120 物理帧。最新灰度图与堆帧输入逐像素核对；样本见 `data/observation_samples.npz`。 |
| 训练曲线 | `logs/dqn_custom_fixed_angle_{500k,refine_200k,recovery_100k,retention_100k}/` 中的 `train.monitor.csv` 和 `eval/evaluations.npz`。四阶段分别计步；两项恢复实验从续训最佳节点分支。节点数值见 `data/training_sources.json`。 |
| 探索率与损失 | 基础训练和续训的 `progress.csv`，不插补缺失损失记录；损失中位数按 20 个日志点平滑。导出数据与来源见 `data/exploration_*.csv`、`data/loss_*.csv` 和 `data/diagnostic_sources.json`。 |
| 得分分布 | 两个已有模型各运行 30 局，逐局种子均为 42000–42029；未更新参数。逐局 CSV、环境契约、软件版本和统计分别位于 `results/evaluation/report_best_30/` 与 `report_final_30/`。配对表见 `data/paired_scores.csv`。 |
| 低分风险与配对差 | 同一组真实逐局分数的经验累积分布和配对差；分位数采用线性插值。Bootstrap 按完整种子对重采样 20000 次、随机数种子 20261017。统计与方法参数见 `data/robustness_statistics.csv` 和 `data/robustness_summary.json`。 |
| 行为连续帧 | 推荐最佳模型在种子 42000 下同一局的两个片段，间隔 12 个物理帧（0.2 秒），只裁取下方接球区域。完整原始帧见 `data/behavior_*.png`；位置、速度、分数、生命和模型 SHA-256 见 `data/rollout_metadata.json`。 |

两个模型分别为：

```text
models/dqn_custom_fixed_angle_refine_200k/best_model/best_model.zip
models/dqn_custom_fixed_angle_refine_200k/final_model.zip
```

复测最佳模型均分 260.2667、中位数 257.00；续训最终模型均分 122.40、中位数 101.50。最佳模型胜出 28/30 局，平均配对差为 137.8667 分，复现原有开发诊断汇总。所有图片均基于实际观测或日志绘制，未从汇总均分反推逐局分数。

补充统计：最佳模型低于 50 分为 1/30 局，续训最终模型为 9/30 局；第 25 百分位得分分别为 210.00 和 36.75。种子 42001、42013 上最佳模型分别少得 198、9 分。当前配对差均值的 Bootstrap 95% 区间为 [95.77,177.43] 分，只反映固定模型在开发样本上的条件变异。

**这仍是开发复测，不是新的最终留出测试，也没有重新训练模型。**

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

绘图脚本默认重建训练、分布、诊断、风险、预处理与行为图；可用 `--only training|distribution|diagnostics|robustness|rollout` 单独重建。中文绘图使用本机 Noto Sans CJK 字体。

网页截图需要 Firefox 和可访问本地端口的环境。先在一个终端运行服务，再在另一个终端截图；脚本会启动模型、等待得分、暂停截图，随后停止本次演示并退出浏览器：

```bash
"$PY" src/dashboard.py --port 8502
# 另一个终端：
/data/zhs/game/.venv/bin/python report/capture_dashboard.py
```

同步 Word：

```bash
"$PY" report/update_word_report.py
```

编译 PDF（需要 TeX Live 中文支持和 Fandol 字体）：

```bash
cd report/latex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
cp build/main.pdf 基于DQN的Breakout课程实验报告.pdf
```

本机系统 TeX 缺少 `ctexart`；本次把 Ubuntu 的 `texlive-lang-chinese_2019.20200218-1_all.deb` 解包到 `/tmp/breakout-report-tex`，未安装系统包。可在上述两次编译命令前加上以下环境变量使用临时依赖：

```bash
TEXINPUTS=/tmp/breakout-report-tex/usr/share/texlive/texmf-dist/tex//: \
OPENTYPEFONTS=/tmp/breakout-report-tex/usr/share/texlive/texmf-dist/fonts/opentype//: \
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
```

临时依赖目录不属于报告交付内容；在完整 TeX Live 环境中直接编译即可。
