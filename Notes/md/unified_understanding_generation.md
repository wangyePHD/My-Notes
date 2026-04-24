# 统一理解与生成（Unified Understanding & Generation）面试笔记

> 面试速查笔记 · 2026/04/11
> 已有基础：BAGEL MoT 架构（代码级）、Flow Matching / DDPM / DDIM 原理、BLIP3o 有所了解
> 目标：建立整个领域的技术版图，掌握流派分类、代表模型、核心设计差异

---

## 🗺️ 知识地图：你在哪里，要去哪里

```
[你已掌握 ✓]
  ├── BAGEL: MoT 架构 + 双支路权重 + Flow Matching 生成
  ├── Flow Matching / DDPM / DDIM 的原理与区别
  └── BLIP3o: 知道存在，有些了解

[本次笔记覆盖 ★]
  ├── 整个领域的"为什么要统一"动机
  ├── 四大技术流派及代表模型
  ├── 视觉表示的核心选择：离散 vs 连续
  ├── 架构分类：共享主干 vs 解耦编码
  └── 主要模型横向对比（Chameleon / Transfusion / Janus / Show-o / BAGEL / BLIP3o）

[进阶方向 →]
  ├── 视频统一模型（Wan / CogVideoX-I2V）
  ├── RL 在统一模型中的应用（Flow-GRPO、UniGRPO）
  └── Any-to-Any 生成（NExT-GPT、UnifiedIO2）
```

---

## 一、为什么需要"统一"？

### 背景动机

传统做法：**理解**和**生成**是两套完全独立的系统。

```
理解侧：图片 → ViT Encoder → LLM → 文本答案
生成侧：文本 → CLIP 文本编码 → UNet（SD/FLUX）→ 图片
```

**问题**：
1. 两套模型维护成本高、推理成本双倍
2. 理解能力无法增强生成（描述不好的模型也生成不好）
3. 无法做真正的**交织多模态**（文字和图片穿插输入输出）
4. 知识共享受限：图像语义理解与生成应当相辅相成

### 统一的核心挑战

> "统一"听起来直觉上很简单——把两件事塞进一个模型。但这里有一个根本矛盾：

| 任务 | 需要什么 |
|------|---------|
| 图像**理解** | 提取**语义**特征（SigLIP/CLIP 风格，连续高维，不可逆压缩） |
| 图像**生成** | 重建**像素**细节（VAE 风格，保留纹理，可解码） |

**同一个视觉表示既要理解语义又要重建像素**——这两个目标天然冲突。

各流派的核心分歧，本质上就是在**怎么解决这个矛盾**。

---

## 二、核心维度：如何表示图像？

在深入流派之前，先搞清楚两个关键的设计选择：

### 2.1 视觉表示：离散 vs 连续

```
图片 ──────────────────────────────────────────────┐
      │                                              │
      ▼ 离散 Tokenization                           ▼ 连续 Latent
  VQ-VAE / VQGAN                              VAE (SD 风格)
  8192 个码本索引                              连续向量（如 16×16×4）
  可以直接做 next-token 预测（AR）            需要扩散/Flow Matching 生成
  压缩率高，信息有损较多                       保留细节，但不能直接 AR
  代表：Chameleon, Janus, Show-o              代表：BAGEL, Transfusion, BLIP3o
```

**直觉理解**：
- **离散**：把图片"翻译"成一串有限词表里的单词，LLM 直接把它当文本 token 处理，整个世界只有一个序列。
- **连续**：保留图片的"模拟信号"，生成时需要扩散模型来建模连续分布，不能简单 next-token。

### 2.2 架构选择：共享主干 vs 解耦编码

```
共享主干（Shared Backbone）         解耦编码（Decoupled Encoder）
─────────────────────────         ──────────────────────────────
理解、生成都走同一个 Transformer    理解侧用 SigLIP/CLIP（语义强）
                                   生成侧用 VQVAE/VAE（重建好）
优点：参数共享，知识互通             优点：两侧各司其职，避免冲突
缺点：两个目标相互干扰               缺点：两套编码器占显存，协调训练难
代表：Chameleon, Transfusion,       代表：Janus, Janus-Pro, EMU2
     Show-o, BAGEL(MoT)
```

---

## 三、四大技术流派

### 流派总览

```
统一理解与生成
├── 流派 A：纯 AR（离散 Token，一个序列统治一切）
│   └── Chameleon, LlamaGen, VAR
│
├── 流派 B：AR + 扩散混合（文本 AR，图像 Diffusion/FM）
│   ├── B1 连续 Latent：Transfusion, BAGEL, BLIP3o
│   └── B2 离散 Masked Diffusion：Show-o
│
├── 流派 C：解耦编码（理解 / 生成各用一套编码器）
│   └── Janus, Janus-Pro, EMU2, SEED-X
│
└── 流派 D：纯扩散统一（Diffusion-only）
    └── OmniGen, InstructPix2Pix
```

---

### 流派 A：纯 AR ——"一切都是 Token"

**核心思想**：把图片通过 VQ-VAE/VQGAN **离散化**成一串 index，拼在文本 token 后面，用同一个 Transformer 做 next-token prediction。

#### 代表：Chameleon（Meta, 2024）

```
[用户] "生成一只猫坐在沙发上"
  ↓
文本 token: [35, 821, 19, ...]
图像 token: [4521, 89, 2234, ...]（VQ-VAE 码本 index，共 1024 个）
  ↓
同一个 LLM，预测下一个 token（文本或图像都行）
  ↓
图像 token 序列 → VQ-VAE 解码器 → 像素
```

**训练目标**：统一的交叉熵损失，无论 token 是文字还是图像 index
$$\mathcal{L} = -\sum_{i} \log P(x_i | x_{<i})$$

**优点**：
- 架构极简，完美的"统一"——真正只有一套模型
- 可以做真正的交织生成（文字中随时插图，图中随时接文字）
- 借助 LLM scaling law

**缺点**：
- VQ-VAE 离散化会**丢失高频细节**，生成图像质量天花板低于扩散模型
- 图像 token 数量多（1024+ per image），序列超长，计算量爆炸
- 词表从 32k 暴增到 32k + 8k（图像码本），tokenizer 需要重设计
- 理解和生成共用同一个 visual tokenizer，两者目标冲突

**关键技术细节**：
- Chameleon 专门设计了 **QK-Norm** + 稳定训练策略，防止图文混合序列训练崩溃
- 图像 VQ-VAE 码本大小通常 8192，分辨率 512×512 对应 1024 个 token

#### VAR（Visual AutoRegressive, ICLR 2024 best paper）

不是统一模型，但影响了后续设计：

```
传统 AR：像素/patch 按光栅扫描顺序，token by token
VAR：coarse-to-fine，先生成 1×1 全局 token，再 2×2，再 4×4...
     每层预测"下一个分辨率"的所有 token（next-scale prediction）
```

**影响**：证明了 AR 生成可以不按光栅扫描，scale-wise 更符合图像的层次结构。

---

### 流派 B1：AR + 连续扩散混合

**核心思想**：文本是离散的，用 AR（next-token）；图像是连续的，用扩散/Flow Matching。两者在**同一个 Transformer 里**共存，但损失函数不同。

> 这正是 **BAGEL** 和 **Transfusion** 的流派，你对 BAGEL 已经很熟悉，下面重点讲它们之间的异同。

#### 代表 1：Transfusion（Meta, 2024）

**一句话**：同一个 Transformer，文本位置算 AR 交叉熵，图像 latent 位置算扩散 MSE loss，两套 loss 联合训练。

```
输入序列：[文本 token, ..., <image>, latent_t, ..., </image>, 文本 token, ...]
              ↓ AR loss                ↓ Diffusion loss
         -log P(x_i|x<i)         ||ε - ε_θ(z_t, t, context)||²
```

**架构关键**：
- 图像 latent 通过一个轻量级 **U-Net style 的局部卷积层**处理（MOA: Modality-of-Attention）
- 文本 token 保持正常 causal attention
- 图像 token 之间做 **bidirectional attention**（扩散是并行去噪，不需要因果约束）
- 文本和图像 token 之间：文本单向 attend 图像，图像双向 attend 文本

**与 BAGEL 的核心区别**：
| | Transfusion | BAGEL |
|--|--|--|
| 图像生成范式 | DDPM 风格扩散（预测噪声 ε） | **Flow Matching**（预测速度 v） |
| 模型参数 | 单套参数（文本/图像 token 共用 Transformer 层） | **MoT 双套权重**（理解侧/生成侧各一套） |
| 图像注意力 | 局部 U-Net 卷积 + 双向 Attention | 全局 Attention，MoT 控制参数分叉 |
| 理解侧 visual encoder | 无（图像直接 patchify 进 latent） | **SigLIP**（独立 ViT，冻结） |
| 训练时序 | 文本 AR + 图像扩散联合训练 | 同上，但有 Flow Matching 时间步 |

#### 代表 2：BAGEL（你已深入了解）

> 关键复习：BAGEL 的核心创新是 **MoT（Mixture of Transformers）**——同一层里，文本 token 走理解支路（无后缀），图像 latent 走生成支路（`moe_gen`），一次 Attention 完成跨模态交互。

参见你的 `bagel.tex` 笔记，这里只补充它在整个流派中的定位：

```
Transfusion：单套参数，试图让同一套权重同时理解和生成
    ↓ 发现两个目标互相干扰
BAGEL：物理分开两套权重（MoT），但共享一次 Attention
    → 理解侧专心学理解，生成侧专心学生成
    → 跨模态语义通过 Attention 自然流通
```

**BAGEL 的完整技术栈**：
```
图像理解侧：图片 → SigLIP ViT → patch embed → LLM (理解支路)
图像生成侧：VAE latent (带噪) → MLP projector → LLM (生成支路)
文本：tokenizer → embedding → LLM (理解支路)
生成输出：LLM (生成支路) → MLP projector → VAE latent → VAE 解码 → 图片
训练目标：理解侧 AR 交叉熵 + 生成侧 Flow Matching MSE
```

#### 代表 3：BLIP3o（Salesforce, 2024）

**定位**：偏向图像生成质量的统一模型，继承 BLIP 系列的理解强项，加入 Flow Matching 生成。

**核心架构**：
```
理解侧：图片 → SigLIP / EVA-CLIP → Q-Former or MLP Projector → LLM
生成侧：文本 → LLM → 隐层特征 → Flow Matching Decoder → 图片
```

**关键技术**：
- **Diffusion Head / Flow Matching Head**：独立的扩散解码器，接收 LLM 最后一层的条件向量
- 与 BAGEL 不同：图像 latent **不进入** LLM 序列里做联合建模，而是 LLM 先编码条件，扩散头单独生成
- 更像："LLM as Text Encoder + 独立的 Diffusion Decoder"

```
BAGEL：[文本 token, 图像 latent token] 在同一个 LLM 序列里 → 更深的融合
BLIP3o：LLM 处理文本/理解 → 输出条件向量 → 独立扩散头生成 → 更模块化
```

**BLIP3o 的理解能力**：继承 BLIP 系列，图像 VQA / 描述能力强，是它的相对优势。

**面试要点**：BLIP3o 代表了一类"轻度统一"的思路——LLM 主体不动，通过插拔生成头来加入生成能力，实现上更简单，但理解与生成的交互深度不如 BAGEL/Transfusion。

---

### 流派 B2：AR + 离散掩码扩散

#### 代表：Show-o（NUS, 2024）

**一句话**：文本用 AR（自回归），图像用 **Masked Diffusion Language Model（MDLM）**——同一个 Transformer，两种 loss。

**关键设计**：
```
文本 token：causal autoregressive，从左到右预测
图像 token (VQ-VAE 离散)：masked diffusion，随机 mask 掉图像 token，预测被 mask 的位置
                           → 类似 BERT 的 MLM，但有时间步 t 控制 mask 比例
```

**与 Chameleon 的区别**：
- Chameleon：图像也是 AR（从左到右）
- Show-o：图像是**双向**的 masked diffusion（所有位置并行预测，不是序列生成）

**与 Transfusion/BAGEL 的区别**：
- Show-o 的图像 token 仍是**离散**的（VQ-VAE index）
- Transfusion/BAGEL 用**连续** latent + 扩散

**直觉**：Show-o 是"离散 token 界的妥协方案"——不想 Chameleon 那样让图像也做 AR（太慢，质量差），但又不想引入连续扩散（太复杂），于是用离散掩码扩散——速度快、并行、还能用码本的整数表示。

---

### 流派 C：解耦视觉编码

**核心动机**：与其让一个 visual tokenizer 同时服务理解和生成（根本矛盾），不如**两件事各用各的工具**。

#### 代表：Janus（DeepSeek, 2024）& Janus-Pro

```
输入图片（理解时）→ SigLIP ViT（语义强，压缩激进）→ LLM
输出图片（生成时）→ LLM → VQ-VAE decoder（细节保留好）→ 图片
              ↑
          同一个 LLM 主干
```

**关键设计**：
- 理解时走 **SigLIP**（高语义，4×4 下采样，每张图 ~256 token）
- 生成时走 **VQVAE**（高保真，用于重建，1024 token per image）
- 两套编码器共享同一个 LLM Backbone
- 没有像 BAGEL 那样做双套权重——LLM 参数共享，但输入来源不同

**Janus-Pro 改进**：
- 更大的图像 tokenizer（码本 16384 vs 8192）
- 更多样的训练数据
- 训练策略三阶段：纯理解 → 纯生成 → 联合微调

**与 BAGEL 的本质区别**：

| | Janus | BAGEL |
|--|--|--|
| 理解侧编码 | SigLIP（进 LLM 的 Q/K/V 统一参数） | SigLIP（进 LLM 的**理解支路**专属参数） |
| 生成侧编码 | VQVAE 离散 token，AR 预测 | VAE 连续 latent，Flow Matching |
| LLM 内部 | **共享**一套参数（理解/生成 token 共用同一套 Q/K/V） | **双套**参数（MoT，理解/生成各一套） |
| 生成质量 | 受限于 VQ-VAE 离散化 | 连续 latent + FM，质量更高 |
| 理解干扰 | 理解/生成共用 LLM 参数，有干扰 | MoT 分支，减少干扰 |

**直觉**：Janus 的解耦只做了"一半"——输入侧解耦了（不同编码器），但 LLM 内部仍是共享参数，理解和生成还是会抢参数空间。BAGEL 的 MoT 把解耦做到了 LLM 内部每一层。

---

### 流派 D：纯扩散统一

#### 代表：OmniGen（2024）

**思路**：不用 AR 语言模型，直接用一个扩散模型处理**所有输入**（包括文字 condition 和图像 condition），输出图像。

```
任意输入（文字 + 图片 + 指令）→ Encoder → 条件向量
         ↓
 DiT (Diffusion Transformer) → 去噪 → 输出图像
```

**适用场景**：图像编辑、风格迁移、条件生成——一个模型搞定。

**局限**：输出只有图像，不能输出文字，所以"理解"能力弱（没有文字输出接口）。

---

## 四、主流模型横向对比表

| 模型 | 机构 | 图像表示 | 生成范式 | LLM 参数 | 理解编码器 | 架构特点 |
|------|------|---------|---------|---------|---------|---------|
| **Chameleon** | Meta | 离散 VQ | AR next-token | 共享 | 无独立编码器（VQ-VAE 做理解） | 极简统一，质量受限 |
| **Transfusion** | Meta | 连续 latent | DDPM 扩散 | 共享（单套） | 无独立 ViT | 文本 AR + 图像扩散，联合 loss |
| **Show-o** | NUS | 离散 VQ | Masked Diffusion | 共享 | 无（VQ token 直接输入） | AR 文本 + 掩码扩散图像 |
| **Janus** | DeepSeek | 离散 VQ | AR next-token | 共享 | SigLIP（理解侧专用） | 解耦输入编码器 |
| **Janus-Pro** | DeepSeek | 离散 VQ | AR next-token | 共享 | SigLIP | Janus 升级版，更大码本 |
| **BAGEL** | 字节 | 连续 VAE | **Flow Matching** | **MoT 双套** | SigLIP（冻结 ViT） | MoT，文本 AR + latent FM |
| **BLIP3o** | Salesforce | 连续 VAE | Flow Matching | 共享（外挂 FM 头） | SigLIP / EVA-CLIP | LLM 输出条件 → 独立扩散头 |
| **EMU2** | BAAI | 连续 | AR + 扩散 | 共享 | EVA-CLIP | 多模态交织生成 |
| **OmniGen** | - | 连续 | 纯 DiT 扩散 | 无 LLM | - | 全扩散，无文字输出 |

---

## 五、视觉 Tokenizer 深挖（面试常考）

### 5.1 VQ-VAE vs VAE：两套逻辑

```
VQ-VAE（离散）：
图片 → Encoder → 连续向量 → 最近邻码本查找 → 离散 index
               ←─────────────────────────────────── 训练反传（straight-through estimator）
离散 index → 码本 embed → Decoder → 重建图片

VAE（连续）：
图片 → Encoder → μ, σ → 采样 z = μ + σε
z → Decoder → 重建图片
训练：重建 loss + KL 散度
```

**关键区别**：
- VQ-VAE 输出整数 index（可以直接被 LLM 的 vocab 处理）
- VAE 输出连续向量（不能 next-token prediction，必须用扩散）

### 5.2 SigLIP 为什么只用于理解不用于生成？

SigLIP（Sigmoid Loss Image-Language Pretraining）是 OpenCLIP 的改进版：
- 训练目标：让图文对的 sigmoid 相似度高，非对应的低
- 输出：高度**语义化**的 embedding，丢掉了像素级细节
- 压缩激进：一张 224×224 图片 → 196 个 patch token（14×14）

**对比 VAE（用于生成）**：
- VAE 训练目标是**重建像素**，保留高频细节
- 一张图 → 16×16 = 256 个 latent，每个是 4/8/16 维连续向量

所以：
```
SigLIP → 告诉你"图里有猫" → 适合理解（VQA、Caption）
VAE    → 告诉你"猫的每根毛的颜色" → 适合生成（重建）
```

BAGEL、Janus 都用 SigLIP 做理解侧，VAE/VQVAE 做生成侧，正是认识到了这个根本区别。

---

## 六、训练策略：统一模型怎么练？

### 6.1 多阶段训练（几乎所有模型都用）

```
阶段 1（预训练基础）：大规模图文对，分别练理解和生成，避免早期干扰
  ↓
阶段 2（联合训练）：理解任务 + 生成任务同时训练，loss 加权求和
  ↓
阶段 3（指令微调 SFT）：高质量指令数据，对齐人类意图
  ↓
阶段 4（可选 RL 微调）：GRPO/DPO，进一步提升生成质量（如 BAGEL 的 Flow-GRPO）
```

### 6.2 Loss 平衡是核心挑战

**问题**：
- 理解任务的 loss（交叉熵）和生成任务的 loss（Flow Matching MSE）量级不同
- 如果直接加权，某一侧会 dominate，另一侧 loss 退化

**常见做法**：
- **Loss 归一化**：分别对两侧 loss 归一化到同一量级
- **Gradient surgery**：梯度冲突时投影，避免两侧任务互相破坏
- **Batch ratio**：控制理解样本 vs 生成样本的比例
- **训练阶段分离**：早期只练一侧，稳定后才联合

### 6.3 Classifier-Free Guidance（CFG）在统一模型中

生成图片时通常用 CFG 提升质量：
```
ε_guided = ε_uncond + w × (ε_cond - ε_uncond)
```

统一模型的 CFG 实现：
- **条件**：文本描述（通过 LLM 编码）
- **无条件**：用空字符串/null token 作为条件
- BAGEL 的 `InterleaveInferencer` 中维护两份 context（有条件 + 无条件），CFG 在 Flow Matching 去噪时应用

---

## 七、面试高频问题 & 答题框架

### Q1：统一理解与生成的核心挑战是什么？

**答题框架**：
> 核心矛盾是**视觉表示的双重需求**：理解需要语义压缩（SigLIP 风格），生成需要像素重建（VAE 风格）。各流派的核心分歧就是怎么解决这个矛盾：
> - Chameleon/Janus 选**两套编码器**（VQ 理解 + VQ 生成或 CLIP + VQ）
> - BAGEL 选**两套编码器 + MoT 双套 LLM 权重**（把分离推进到 LLM 内部）
> - Transfusion 选**一套权重**但接受两侧相互干扰，用联合 loss 平衡
> - BLIP3o 选**外挂生成头**，最小侵入 LLM 主体

### Q2：BAGEL 和 Janus 都用了 SigLIP，有什么本质区别？

**答**：
- Janus：SigLIP patch 进入 LLM 后，理解和生成 token 共用**同一套** Q/K/V/MLP，LLM 内部没有分离
- BAGEL：SigLIP patch 进入 LLM 的**理解支路**（无后缀参数），图像 VAE latent 进入**生成支路**（`moe_gen` 后缀参数），同一层内两套权重，但一次 Attention 互通

本质区别：BAGEL 的分离是**参数级**的（每层都有两套），Janus 的分离只是**编码器级**的（LLM 以前分，LLM 里共用）。

### Q3：为什么 BAGEL 用 Flow Matching 而不是 DDPM？

**答**：
- Flow Matching 路径是线性插值（直线），DDPM 是弯曲随机游走
- 训练更稳定：FM 的目标 $v = x_1 - x_0$ 是常数方向，不同时间步的 loss 量级均匀
- 采样更快：直线路径让 ODE 积分步数更少（与 FLUX/SD3 同一技术路线）
- BAGEL 联合 LLM 训练时，训练稳定性很重要，FM 的均匀 loss 有助于多任务平衡

### Q4：离散 Token（VQ-VAE）和连续 Latent（VAE）各有什么优劣？

| | 离散 VQ-VAE | 连续 VAE |
|--|--|--|
| 与 LLM 的兼容性 | 天然兼容（整数 index） | 需要额外扩散模型 |
| 生成质量上限 | 受码本大小限制，有量化误差 | 更高，无量化损失 |
| 生成速度 | AR 逐 token 慢（1024+ 步） | 扩散并行，通常更快 |
| 训练复杂度 | 统一 loss（交叉熵），简单 | 多种 loss，需平衡 |
| 代表模型 | Chameleon, Janus, Show-o | BAGEL, Transfusion, BLIP3o |

**趋势**：工业界高质量生成正在转向**连续 latent + Flow Matching**（与 FLUX/SD3 同轨）。

### Q5：统一模型如何处理图像理解中的"多图输入"问题？

**答**：以 BAGEL 为例：
- 每张图片通过 SigLIP ViT 提取 patch features（约 256-1024 个 token/图）
- 通过 MLP projector 映射到 LLM 的隐藏维度
- 拼入序列，做 `forward_cache_update_vit`（更新 KV cache）
- 多张图片顺序缓存，形成长上下文，LLM 统一理解

挑战：多图会导致序列极长（4 张图 × 1024 token = 4096 仅用于图像），需要 flash attention 或 sequence packing。

### Q6：Chameleon 训练会不稳定，为什么？怎么解决？

**答**：
- 原因：文字 token 和图像 token 的 embedding 分布差异大，联合训练时梯度冲突，导致 loss spike 甚至训练崩溃
- Chameleon 的解决方案：**QK-Norm**（对 Attention 的 Query 和 Key 做 RMSNorm），控制内积量级，稳定 softmax
- 此外：**z-loss**（对 logits 做正则），防止某些 token 概率接近 1

### Q7：什么是 MoT（Mixture of Transformers），它解决了什么问题？

**答**（直接从你的 bagel.tex 整理）：
> MoT 是 BAGEL 的核心架构创新。传统统一模型让理解和生成 token 共用同一套参数，导致两个目标互相干扰。MoT 在每个 Transformer 层内物理分为两套参数（理解支路无后缀，生成支路 `moe_gen`），但**只做一次 Attention**——两侧 token 在注意力中仍然互相可见，语义交流没有割断。
>
> 三要素：一条序列（文本 + 图像 latent 打包）、两套权重（按 token 类型切换）、一次 Attention（全局互通）。

---

## 八、知识体系图：从 BAGEL 出发看整个领域

```
                    统一理解与生成
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    视觉表示         LLM 架构        生成范式
    离散 vs 连续     共享 vs 解耦    AR vs 扩散 vs 混合
          │              │              │
     ┌────┴────┐    ┌────┴────┐    ┌────┴──────┐
   离散       连续  共享参数  解耦  纯AR   扩散  混合
  Janus    BAGEL  Chameleon Janus  Chameleon Transfusion Show-o
  Show-o   BLIP3o Transfusion     LlamaGen  OmniGen  BAGEL(MoT)
  Chameleon       Show-o                             BLIP3o

你的位置：BAGEL（连续 + 解耦MoT + 混合Flow Matching）
          理解最深的模型也是最前沿的设计之一 ✓
```

---

## 九、进阶方向与推荐资源

### 下一步可以学习的方向

1. **Janus/Janus-Pro 论文**（DeepSeek 2024）：与 BAGEL 对比学习，加深对"解耦"策略的理解
2. **Transfusion 论文**（Meta 2024）：理解"同一套参数"统一的方案与 MoT 的权衡
3. **Flow-GRPO / UniGRPO**：你已有 `unigrpo.tex` 笔记，RL 在统一模型中的应用

### 推荐论文（按重要性排序）

| 论文 | 时间 | 核心贡献 |
|------|------|---------|
| Chameleon | 2024.05 | Meta 首个真正统一的 AR 多模态模型 |
| Transfusion | 2024.08 | 文本 AR + 图像扩散，联合训练范式 |
| Janus / Janus-Pro | 2024.10 / 2025.01 | 解耦视觉编码，DeepSeek |
| Show-o | 2024.09 | AR + 离散掩码扩散混合 |
| BAGEL | 2025 | MoT + Flow Matching，你的主战场 ✓ |
| BLIP3o | 2024 | LLM + 独立扩散头，Salesforce |
| VAR | 2024 | Next-scale AR，ICLR best paper |

### 一个面试前的思考题

> **BAGEL 和 Transfusion 都是"文本 AR + 图像扩散"的混合范式，但 BAGEL 用了 MoT 双套权重，Transfusion 用了单套权重。从理论上讲，哪种设计更好？从实际效果上怎么判断？**
>
> 提示：思考"任务冲突"（task conflict）与"知识共享"（knowledge sharing）的权衡，以及如何用消融实验验证 MoT 的必要性。

---

*笔记整理于 2026/04/11 | 基于 bagel.tex / diffusion_paradigms_interview.md 整合扩展*
*适用于：多模态方向研究员面试、实习答辩、组会汇报*
