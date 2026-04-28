# arXiv & Daily Papers 笔记专区

这个目录用于沉淀每天从 arXiv new submissions 和 Hugging Face Daily Papers 中筛出的论文。

## 收录范围

- 后训练：LLM/VLM/RLVR/GRPO/DPO/RLHF/奖励模型/偏好对齐。
- AIGC：视频生成与编辑、图像生成与编辑、统一理解与生成模型。
- 世界模型：交互式视频世界模型、机器人 world model、3D/4D world generation。
- 具身智能：VLA、机器人策略、仿真环境、空间记忆、安全评测。
- Style：图像风格、文本 voice/style、身份保持、设计工作流。

## 文件约定

- 每日总览：`daily_YYYY_MM_DD.tex`
- 单篇深读：`paper_<short_name>_YYYY_MM_DD.tex`
- 入口索引：`index.tex`
- 可复制模板：`daily_template.tex`

新增每日笔记后，在 `index.tex` 里加入对应 `\input{...}`。

## 笔记结构

建议每篇日报固定包含：

- `今日重点`
- `按方向归类`
- `值得深读`
- `暂时略过`
- `后续动作`

这样可以避免只堆论文标题，而是形成可复盘的研究雷达。
