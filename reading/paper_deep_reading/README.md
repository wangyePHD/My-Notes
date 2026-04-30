# Paper Deep Reading

这里放单篇论文精读笔记。精读专区按方向拆成多个小专区，避免所有单篇笔记堆在同一层目录。

定位：

- `arxiv_daily_papers/` 负责每日发现和粗筛。
- `paper_deep_reading/` 负责单篇论文的结构化精读。
- `surveys/` 负责把一组论文整理成专题知识地图。

推荐命名：

```text
topic_folder/YYYY_MM_DD_short_name.tex
```

例如：

```text
post_training/2026_04_27_udm_grpo.tex
```

新增一篇精读笔记后，在 `index.tex` 中加入：

```tex
\input{reading/paper_deep_reading/topic_folder/YYYY_MM_DD_short_name}
```

可复制模板：

```text
paper_template.tex
```

## 当前小专区

```text
paper_deep_reading/
├── post_training/       # 后训练、RLHF/RLVR/GRPO/DPO、奖励模型、信用分配
├── image_models/        # 图像生成、图像编辑、视觉文本生成、可控生成
├── video_models/        # 视频生成、视频编辑、长视频模型、物理一致性
├── unified_models/      # 统一理解与生成、any-to-any、多模态生成
├── world_models/        # 世界模型、3D/4D、空间智能、仿真
├── embodied_ai/         # VLA、机器人策略、具身后训练、仿真环境
├── style_identity/      # 风格、身份保持、设计工作流、审美奖励
├── evaluation_safety/   # 评测、安全、红队、provenance、failure discovery
└── systems_inference/   # 训练/推理系统、attention、采样加速、工程优化
```

每个子目录有自己的 `index.tex`。新增论文时：

1. 把 `.tex` 放入最主要方向的子目录。
2. 在该子目录的 `index.tex` 中添加 `\input{...}`。
3. 如果论文跨方向，在其它子目录的 `index.tex` 里用一句交叉引用说明，不重复全文。
