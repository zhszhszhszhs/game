# 强化学习课程实验报告

正文、封面、公式、表格、参考文献及图片占位逻辑均位于 `main.tex`。

## 编译

选择 XeLaTeX，连续编译两次以更新目录和交叉引用：

```sh
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

需要带中文支持的 TeX Live 或 MiKTeX。源码使用 TeX 发行版自带的 Fandol 字体，不依赖本机商业字体。也可上传到 Overleaf 并选择 XeLaTeX。

## 封面

在源码开头修改 `MajorClass`、`StudentID`、`StudentName`、`Instructor` 等命令。封面参考所提供报告的布局，个人信息保留待填写，课程名称为“强化学习”。

## 图片替换

将图片放入 `images/`，文件存在后会自动替换相应占位框：

| 文件 | 内容 |
| --- | --- |
| `game_live.png` | 网页中的游戏实时展示 |
| `observation_pipeline.png` | 原始画面、灰度缩放与四帧堆叠 |
| `training_curves.png` | 训练回合奖励和开发评估曲线 |
| `score_distribution.png` | 相同开发种子的配对得分或分布 |
| `behavior_sequence.png` | 成功接球与失球过程的连续帧 |

可以在 `reportfigure` 调用中修改路径以使用 PDF、JPG 等格式。替换后同步将图注中的“待补充”删除，并依据实际图片修改分析文字。缺少所有图片时，源码也可以完整编译。

实验数值来自项目的现有记录，文中区分开发评估与最终留出测试。未记录的实验统计量没有填入推测值。
