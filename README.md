# My Notes

这是一个面向长期写作的 LaTeX 笔记仓库：本地写作，GitHub 保存版本，Overleaf 使用 `main.tex` 编译。

## 目录层级

```text
.
├── main.tex                  # Overleaf 编译入口，只负责加载样式和正文索引
├── tex/
│   └── preamble.tex          # 全局宏包、颜色、样式、快捷命令
├── Notes/
│   ├── index.tex             # 正文编译顺序；新增笔记后在这里加 \input
│   ├── 00_learning_system/   # 学习系统：新领域学习法、模板、专题学习路线、学习对话记录
│   ├── 01_foundations/       # 基础理论：Diffusion / Flow Matching 等
│   ├── 02_model_architectures/# 模型架构：FLUX / BAGEL / Z-Image 等
│   ├── 03_rl_post_training/  # RL 后训练：GRPO / Flow-GRPO / UniGRPO 等
│   ├── 04_systems/           # 系统与推理：KV Cache / 分布式训练等
│   ├── 05_interviews/        # 面试复盘
│   └── md/                   # Markdown 版本或导出稿
├── assets/
│   └── figures/              # 图片资源；LaTeX 中可直接写文件名
├── bib/
│   └── references.bib        # 参考文献库
├── reading/
│   ├── arxiv_daily_papers/   # arXiv & Hugging Face Daily Papers 每日论文雷达专区
│   ├── paper_deep_reading/   # 论文精读专区：单篇论文结构化拆解
│   ├── surveys/              # Survey 专区：专题综述、论文地图、学习路线
│   └── arxiv_weekly/         # 历史 arXiv 论文速读，已转为 LaTeX
├── scripts/                  # 辅助脚本
└── latexmkrc                 # 本地使用 XeLaTeX 编译
```

## 新增一篇 LaTeX 笔记

1. 把新文件放到对应领域目录，例如：

```text
Notes/03_rl_post_training/new_method.tex
```

2. 在 `Notes/index.tex` 中加入：

```tex
\input{Notes/03_rl_post_training/new_method}
```

3. 提交并推送：

```bash
git add .
git commit -m "Add new method note"
git push
```

## 新增一轮学习对话

每个专题学习放在一个 session 子目录，例如：

```text
Notes/00_learning_system/study_sessions/post_training_2026_04_24/
```

每一轮对话单独写一个 `.tex`：

```text
round_02_reward_model.tex
```

然后在该 session 的 `index.tex` 中加入：

```tex
\input{Notes/00_learning_system/study_sessions/post_training_2026_04_24/round_02_reward_model}
```

## 新增一篇 arXiv / Daily Papers 笔记

每日筛选和总览统一放在：

```text
reading/arxiv_daily_papers/
```

推荐命名为：

```text
daily_YYYY_MM_DD.tex
```

然后在专区入口里加入：

```tex
\input{reading/arxiv_daily_papers/daily_YYYY_MM_DD}
```

专区入口文件是：

```text
reading/arxiv_daily_papers/index.tex
```

单篇深读可以命名为：

```text
paper_short_name_YYYY_MM_DD.tex
```

历史周报仍保留在：

```text
reading/arxiv_weekly/
```

可复制模板：

```text
reading/arxiv_daily_papers/daily_template.tex
```

如果先写成 Markdown，可以运行转换脚本：

```bash
python3 scripts/convert_arxiv_md_to_tex.py
```

脚本会把 `reading/arxiv_weekly/*.md` 转成同名 `.tex`，并删除原 `.md`。

## 新增一篇论文精读笔记

单篇论文精读统一放在：

```text
reading/paper_deep_reading/
```

推荐命名为：

```text
YYYY_MM_DD_short_name.tex
```

然后在专区入口里加入：

```tex
\input{reading/paper_deep_reading/YYYY_MM_DD_short_name}
```

专区入口文件是：

```text
reading/paper_deep_reading/index.tex
```

可复制模板：

```text
reading/paper_deep_reading/paper_template.tex
```

## 新增一个 Survey 专题

专题综述统一放在：

```text
reading/surveys/
```

每个专题单独建目录，例如：

```text
reading/surveys/discrete_diffusion/
```

专题入口文件是：

```text
reading/surveys/discrete_diffusion/index.tex
```

然后在 Survey 总入口里加入：

```tex
\input{reading/surveys/discrete_diffusion/index}
```

Survey 总入口文件是：

```text
reading/surveys/index.tex
```

## 编译约定

- 使用 `main.tex` 作为唯一入口文件。
- 使用 XeLaTeX 编译，中文支持更稳定。
- 图片统一放在 `assets/figures/`，因为 `tex/preamble.tex` 已设置 `\graphicspath`，正文里可以直接写：

```tex
\includegraphics[width=\linewidth]{example.png}
```
