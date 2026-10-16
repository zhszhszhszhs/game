# 强化学习课程实验报告

正文、封面、公式、表格及图片均位于 `main.tex`。2026-10-17 已补齐七张实验图片，并同步更新图注、分析及 PDF；数据来源和复现命令见 [上级说明](../README.md)。

## 编译

选择 XeLaTeX，连续编译两次以更新目录和交叉引用：

```sh
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

需要带中文支持的 TeX Live 或 MiKTeX。源码使用 TeX 发行版自带的 Fandol 字体，不依赖本机商业字体。也可上传到 Overleaf 并选择 XeLaTeX。

## 封面

在源码开头修改 `MajorClass`、`StudentID`、`StudentName`、`Instructor` 等命令。封面已填写硕2605班、M202674292、赵恒升、何琨，课程名称为“强化学习”。

## 已补齐的图片

图片位于 `images/`，全部来自本地项目日志或实际运行：

| 文件 | 内容 |
| --- | --- |
| `game_live.png` | Firefox 截取实际网页，推荐最佳模型，种子 42000，得分 20、生命 5 |
| `observation_pipeline.png` | 环境回调采集 RGB、末两帧池化及实际四帧输入，最新灰度帧与网络输入逐像素一致 |
| `training_curves.png` | 四个正式实验的完整 Monitor 日志与开发评估 NPZ，标注最佳与早停节点 |
| `training_diagnostics.png` | 基础训练和续训的实际探索率与 Huber 损失，标记开发最佳节点 |
| `score_distribution.png` | 重新运行两个模型各 30 局后的真实配对分数与箱线图 |
| `score_robustness.png` | 同种子得分累积分布、低分风险与排序配对差 |
| `behavior_sequence.png` | 同一模型、同一局的接球与失球片段，标注时间、得分与生命 |

可以在 `reportfigure` 调用中修改路径以使用 PDF、JPG 等格式。替换图片后应同步更新图注和分析文字；源码保留缺失图片时的占位回退逻辑。

文中区分开发评估与最终留出测试。本次复测复现原有均分，并补充实测中位数等逐局统计量；最终留出测试尚未执行。
