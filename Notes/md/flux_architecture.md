# FLUX.1 模型架构详解（面试备战版）

> **Black Forest Labs, 2024.08**  
> 前 Stability AI 核心成员（Stable Diffusion 原班人马）打造

---

## 知识地图

```
[你已掌握 ✓]                [本次讲解 ★]                    [进阶方向 →]
Flow Matching          →   Flow Matching 细节原理       →  Consistency Model
双流/单流架构          →   双流/单流内部结构             →  DiT / PixArt
                       →   RoPE 位置编码                →  2D/3D RoPE
                       →   T5 + CLIP 双编码器           →  Prompt Engineering
                       →   16通道 VAE                   →  Latent Diffusion
                       →   Guidance Distillation        →  LCM / SDXL-Turbo
                       →   QK-Norm + 并行注意力         →  训练稳定性技巧
                       →   三个模型变体的区别            →  部署/推理优化
```

---

## 一、整体架构鸟瞰

FLUX.1 本质上是一个 **12B 参数的 Multimodal Diffusion Transformer（MMDiT）**，在 SD3 的基础上做了大量改进。

```
文本输入
  ├─ CLIP ViT-L (pooled)    ─→ vec (全局语义向量)
  └─ T5-XXL (sequence)      ─→ txt tokens (细粒度语义序列)

图像输入（加噪后的 latent）
  └─ VAE Encoder (16ch)     ─→ img tokens（patchify 成 2×2 patches）

              ↓ 时间步 t + guidance scale → 融入 vec

        ┌─────────────────────────┐
        │  Double Stream Blocks   │  × N（图文各自独立处理，cross-attend）
        └─────────────────────────┘
                    ↓
        ┌─────────────────────────┐
        │  Single Stream Blocks   │  × M（图文 concat，联合处理）
        └─────────────────────────┘
                    ↓
              Linear Head
                    ↓
          VAE Decoder (16ch)
                    ↓
              最终图像输出
```

---

## 二、Flow Matching（流匹配）

### 直觉理解
传统 DDPM 加噪是"弯路"，Flow Matching 学的是从噪声 → 图像的**最短直线路径**（velocity field）。

### 原理
Flow Matching 的训练目标：学习一个速度场 $v_\theta(x_t, t)$，使得从噪声 $x_1 \sim \mathcal{N}(0,I)$ 出发，沿速度场积分能到达真实数据 $x_0$。

FLUX 使用 **Rectified Flow（整流流）**，插值方式为线性：

$$x_t = (1-t) \cdot x_0 + t \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

训练目标是最小化：

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left\| v_\theta(x_t, t) - (\epsilon - x_0) \right\|^2$$

即：**让模型预测的速度方向 = 从数据指向噪声的方向**（即 $\epsilon - x_0$）。

### 对比 DDPM 的优势
| 对比点 | DDPM | Rectified Flow |
|--------|------|----------------|
| 轨迹 | 曲线（随机游走） | 直线 |
| 采样步数 | 通常 1000 步 | 少步即可（20~50，或蒸馏后 1~4 步） |
| 训练目标 | 预测噪声 $\epsilon$ | 预测速度 $v = \epsilon - x_0$ |
| 推导复杂性 | 需要 ELBO 推导 | 直接 OT 对齐，概念更简洁 |

### Inference（推理）过程
使用 ODE solver（如 Euler method）从 $t=1$（纯噪声）积分到 $t=0$（图像）：

$$x_{t - \Delta t} = x_t - \Delta t \cdot v_\theta(x_t, t)$$

---

## 三、双流块（Double Stream Blocks）

### 结构
图像 token 和文本 token **各走各的 Transformer 流**，但通过 Attention 相互影响。

```
输入：
  img_tokens  [B, N_img, D]
  txt_tokens  [B, N_txt, D]
  vec         [B, D]           ← 来自 CLIP + timestep + guidance

每个 Double Stream Block 内部：

  ┌──── Image Stream ────┐    ┌──── Text Stream ────┐
  │ LayerNorm             │    │ LayerNorm            │
  │ Modulation(vec)       │    │ Modulation(vec)      │
  │  ↓ scale/shift/gate   │    │  ↓ scale/shift/gate  │
  │ Self-Attn (img Q,K,V) │    │ Self-Attn(txt Q,K,V) │
  │ ← cross with txt K,V  │    │ ← cross with img K,V │
  │ FFN                   │    │ FFN                  │
  └───────────────────────┘    └──────────────────────┘
```

关键点：
- 两流都用 **RoPE** 做位置编码
- Attention 时 img 和 txt 的 Q/K 拼接后一起算，但各自来自独立权重
- 用 **QK-Norm** 对 Q 和 K 做 LayerNorm（稳定训练）
- **Modulation 机制**：用 `vec`（全局条件向量）预测每个 token 的 scale/shift/gate，而不是用 AdaLN

---

## 四、单流块（Single Stream Blocks）

### 结构
把 img_tokens 和 txt_tokens **直接拼接**，当成一个序列联合处理：

```
tokens = concat(img_tokens, txt_tokens)  → [B, N_img + N_txt, D]

Single Stream Block 内部：

  LayerNorm
  Modulation(vec)   ← 仍用 vec 全局调制
  ↓
  ┌────────── Parallel ──────────┐
  │ Attention head (Q,K,V)       │   ← 图文 tokens 统一做 self-attn
  │ MLP (FFN)                    │   ← 同时并行执行，而非串行
  └──────────────────────────────┘
  ↓ 两路输出相加，再过 linear projection
```

关键点：
- **并行注意力（Parallel Attention）**：Attention 和 FFN 同时计算，减少内存带宽瓶颈，加速推理
- 图文 tokens 完全打通，实现深度融合
- 最终只取 img_tokens 部分过 linear head 输出 velocity

### 双流 vs 单流的直觉区别

| | 双流块 | 单流块 |
|--|-------|-------|
| 位置 | 前几层 | 后几层 |
| 目的 | 各自提取模态特征，再交互 | 深度融合，精细生成 |
| 类比 | 翻译时先各自读懂，再对照 | 双语融合后统一理解 |

---

## 五、RoPE（旋转位置编码）

### 直觉理解
不是给 token 加一个"绝对座位号"，而是让 token 的 Q 和 K 向量**旋转一个角度**，旋转量由位置决定。两个 token 的注意力得分自动编码了它们的**相对距离**。

### 为什么 FLUX 用 RoPE？
图像生成需要处理任意分辨率（512×512、1024×768...），如果用绝对位置编码，换分辨率就崩了。RoPE 只编码相对位置，天然支持**变长、变分辨率**。

### FLUX 的 2D RoPE 实现
图像 patch 有 `(row, col)` 两个维度，FLUX 用分解的 2D RoPE：

```python
# EmbedND：构造位置张量
# 对 (height_ids, width_ids) 分别计算 1D RoPE，拼接成 2D
# apply_rope：把旋转矩阵作用到 Q, K 上
```

这样模型可以感知 patch 的二维空间关系，而不只是序列顺序。

---

## 六、文本编码器：T5-XXL + CLIP

### 为什么要两个？

| 编码器 | 参数量 | 输出类型 | 作用 |
|--------|--------|---------|------|
| **CLIP ViT-L** | ~400M | pooled embedding（全局向量） | 图文对齐，提供整体语义 |
| **T5-XXL** | 11B | sequence tokens（序列） | 理解复杂、长文本，精细控制 |

### 如何使用
- **CLIP pooled** → 加入 `vec`（全局调制向量），影响每个 block 的 Modulation
- **T5 tokens** → 作为 `txt_tokens`，参与双流块的 cross-attention

这解决了 SD1.x 只用 CLIP、对复杂 Prompt 理解差的问题（T5 是语言模型出身，理解自然语言更强）。

---

## 七、16通道 VAE

### 对比
| 模型 | VAE 通道数 | 压缩比 |
|------|-----------|-------|
| SD 1.x / 2.x | 4 | 8× |
| FLUX.1 | **16** | 8× |

FLUX 的 VAE 同样是 8× 空间压缩（1024px → 128 latent），但 latent 有 **16 个通道**而非 4 个，信息密度更高。

### Patchify
图像 latent `[B, 16, H, W]` → 切成 2×2 patches → `[B, N, 16*4=64]`，作为 img tokens 输入 Transformer。

---

## 八、Guidance Distillation（引导蒸馏）

### 背景：CFG 的问题
Classifier-Free Guidance 每步要跑 **两次前向**（条件 + 无条件），然后做插值：

$$\hat{v} = v_\text{uncond} + w \cdot (v_\text{cond} - v_\text{uncond})$$

慢，且存在 artifact。

### FLUX.1 [dev] 的解法：Guidance Distillation
把 guidance scale $w$ 作为**额外输入**，用教师模型（跑 CFG）的输出蒸馏学生模型（只跑一次），让模型**内化** CFG 的效果：

```
时间编码 t  
引导编码 w   →  融合 →  vec  →  Modulation 各个 Block
CLIP 嵌入
```

推理时只需一次前向，guidance scale 通过 `vec` 传入即可（如 `guidance_scale=3.5`）。

### FLUX.1 [schnell] 的解法：Timestep Distillation（LADD）
进一步把 50 步压缩到 **1~4 步**：
- 使用 **Latent Adversarial Diffusion Distillation (LADD)**
- 引入判别器网络，用对抗训练保持质量
- 推理时 `guidance_scale=0`（不需要 guidance 了）
- 代价：max sequence length 256，复杂 Prompt 能力弱一些

---

## 九、三个模型变体对比

| | **FLUX.1 [pro]** | **FLUX.1 [dev]** | **FLUX.1 [schnell]** |
|--|---|---|---|
| 访问方式 | API Only（闭源） | 开放权重（非商用） | 开放权重（Apache 2.0）|
| 蒸馏方式 | 无蒸馏（完整 CFG） | Guidance Distillation | Guidance + Timestep 蒸馏（LADD） |
| 推理步数 | 50+ | 50 | 1~4 |
| guidance_scale | 需要 CFG | 3.5（单次前向） | 0 |
| max seq len | 无限制 | 512 | 256 |
| 质量 | 最好 | 接近 Pro | 略低，但极快 |
| 适用场景 | 商业图像生成 | 研究/高质量本地 | 实时生成/原型 |

---

## 十、训练稳定性技巧

### QK-Norm
在计算 Attention 之前，对 Q 和 K 分别做 LayerNorm：

$$\text{Attention} = \text{softmax}\left(\frac{\text{Norm}(Q) \cdot \text{Norm}(K)^T}{\sqrt{d}}\right) V$$

防止 12B 大模型训练时注意力分数爆炸，梯度不稳定。

### HybridNorm
- **Pre-Norm**：用在 QKV 投影之前（保证 Attention 稳定）
- **Post-Norm**：用在 FFN 之后（防止残差连接引起的输出漂移）

这是大模型训练的 trick，FLUX 在超大参数量下保证了训练稳定。

### 并行注意力（Parallel Attention）
Single Stream Block 中 Attention 和 FFN 并行：

```
x → Norm → [Attn(x) || FFN(x)] → 相加 → Linear
```

相比串行，**计算效率提升**，尤其在推理时减少内存访问延迟。

---

## 十一、Attention 机制全景 & 条件图像注入

### 11.0 FLUX Attention 的三个维度

在读源码之前，先把 FLUX 里 Attention 的"全貌"拎清楚——同一套 `attention()` 函数，因为**输入序列的构成不同**，呈现出三种不同的语义角色：

```
序列组成（双流块）：  [txt tokens | img tokens]
                           ↑             ↑
                    文本语义区        图像空间区
                    txt_ids全0      img_ids含(row,col)

序列组成（单流块）：  [txt tokens | img tokens]  （concat后统一处理）

序列组成（Kontext）： [img_cond tokens | txt tokens | img_noisy tokens]
                          ↑                               ↑
                    条件图（ids第0维=1）           待生成图（ids第0维=0）
```

**关键设计哲学**：FLUX 不用显式 Cross-Attention，而是把所有 token（文本、噪声图、条件图）全部扔进同一个 Self-Attention 序列里，用 **RoPE 的位置 id 来区分它们的"身份"**。想让两个 token 互相看到对方？把它们放在同一序列里就行了。

---

### 11.1 条件图像注入的三种方式（对比）

| 方式 | 代表模型 | 注入位置 | 改没改 Attention | 适用场景 |
|------|---------|---------|----------------|---------|
| **Channel concat**（通道拼接） | Fill（Inpainting）、Canny、Depth | `img` token 的**特征维度**上拼接 | ❌ 不改，只改 `img_in` 的输入维度 | 像素级对齐的条件（mask、边缘图） |
| **Sequence concat**（序列拼接） | Kontext | `img` token 的**序列长度**上拼接 | ✅ 条件图 tokens 完整参与 Attention | 参考图风格/内容迁移 |
| **txt 侧 concat** | Redux | 拼接到 `txt tokens` 序列末尾 | ✅ 通过双流的图文交互影响生成 | 图像风格/内容参考（无需空间对齐）|

---

### 11.2 方式一：Channel Concat（Fill / Canny / Depth）

#### 核心思路
条件图（inpainting mask、边缘图、深度图）与**噪声 latent 逐像素对齐**，直接在通道维度拼接，作为额外输入通道传给 `img_in`。

#### 源码：`prepare_fill()` & `prepare_control()`

```python
# ── prepare_control（Canny / Depth）────────────────────────────────
img_cond = ae.encode(encoder(img_cond))           # 条件图 → VAE latent [B,16,H/8,W/8]
img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
# img_cond shape: [B, N_patch, 64]  与 img 完全相同

return_dict["img_cond"] = img_cond
# ← 只是记录下来，还没 concat！concat 在 denoise() 里发生

# ── prepare_fill（Inpainting）──────────────────────────────────────
img_cond = ae.encode(img_cond * (1 - mask))       # mask 区域置零后编码
mask = rearrange(mask, "b (h ph) (w pw) -> b (h w) (ph pw)", ph=8, pw=8)
# mask 也 patchify，变成每个 patch 的遮盖率

img_cond = torch.cat((img_cond, mask), dim=-1)
# img_cond shape: [B, N_patch, 64+4=68]  ← 多了 mask 通道
```

#### 源码：`denoise()` 里的 channel concat

```python
def denoise(model, img, img_ids, ..., img_cond=None, ...):
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):

        img_input = img                             # 噪声 latent: [B, N, 64]

        if img_cond is not None:
            img_input = torch.cat((img, img_cond), dim=-1)
            # ⭐ 通道维度 concat！
            # Canny/Depth: [B, N, 64] cat [B, N, 64] → [B, N, 128]
            # Fill:        [B, N, 64] cat [B, N, 68] → [B, N, 132]

        pred = model(img=img_input, img_ids=img_ids, ...)
        # img_input 进入 img_in = nn.Linear(in_channels, hidden_size)
        # ← 所以 Canny/Fill 的模型 in_channels 不是 64，而是 128 / 132！
```

#### 对 Attention 的影响

```
Channel Concat 方案：Attention 结构完全不变！
                    序列长度 N 不变
                    只有 img_in 线性层的输入维度变了（64 → 128/132）
                    条件信息通过权重矩阵"混入"每个 patch 的 hidden 表示
                    之后参与正常的双流/单流 Attention

图示：
原始：  img_noisy [B, N, 64]  → img_in(64→hidden) → Attention
Canny： img_input [B, N,128]  → img_in(128→hidden) → Attention（同一套 block，重新训练 img_in）
```

> 💡 这种方式的本质：**让每个 patch 在进入 Transformer 之前就"知道"自己对应位置的条件信息**，是最局部、最对齐的条件注入方式。代价是需要对 `img_in` 重新微调（输入维度变了）。

---

### 11.3 方式二：Sequence Concat（Kontext）

#### 核心思路
条件图像经过 VAE 编码后变成 patch tokens，直接**拼接到噪声图 tokens 的序列前面**，两者在 Attention 中完全平等地互相 attend。用 **RoPE 的第 0 轴（时间轴）区分条件图（id=1）和生成图（id=0）**。

#### 源码：`prepare_kontext()`

```python
# ── 条件图编码 ────────────────────────────────────────────────────
img_cond = ae.encode(img_cond)
img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
# img_cond: [B, N_cond, 64]  与普通 img tokens 格式完全相同

# ── ⭐ 关键：条件图的 img_ids，第 0 维设为 1（区别于生成图的 0）──
img_cond_ids = torch.zeros(height//2, width//2, 3)
img_cond_ids[..., 0] = 1          # ← 时间轴 id = 1（条件帧）
img_cond_ids[..., 1] = torch.arange(height//2)[:, None]   # row
img_cond_ids[..., 2] = torch.arange(width//2)[None, :]    # col
# 生成图的 img_ids[..., 0] = 0（默认）

return_dict["img_cond_seq"]     = img_cond
return_dict["img_cond_seq_ids"] = img_cond_ids
```

#### 源码：`denoise()` 里的 sequence concat

```python
if img_cond_seq is not None:
    img_input     = torch.cat((img_input, img_cond_seq), dim=1)
    # ⭐ 序列维度 concat！
    # [B, N_gen, 64] cat [B, N_cond, 64] → [B, N_gen+N_cond, 64]

    img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
    # ids 也拼接，用于统一算 RoPE
    # [B, N_gen, 3] cat [B, N_cond, 3] → [B, N_gen+N_cond, 3]

pred = model(img=img_input, img_ids=img_input_ids, ...)

# 推理完取回生成图部分（丢掉条件图 tokens 的预测结果）
pred = pred[:, :img.shape[1]]    # 只要前 N_gen 个 token 的输出
```

#### 对 Attention 的影响（这是核心！）

```
Sequence Concat 方案：Attention 序列长度翻倍（约）！

双流块中的联合 Attention 序列：
  原始：  [txt(256) | img_noisy(N_gen)]
  Kontext：[txt(256) | img_noisy(N_gen) + img_cond(N_cond)]
                                          ↑
                               作为 img tokens 的后半段参与

完整的 Attention 矩阵（以双流块为例）：
         txt  |  img_noisy  |  img_cond
  txt  [ ✓✓  |     ✓✓      |    ✓✓   ]  ← 文本看到所有图像
  img  [ ✓✓  |     ✓✓      |    ✓✓   ]  ← 生成图看到条件图 ⭐
 cond  [ ✓✓  |     ✓✓      |    ✓✓   ]  ← 条件图也看到生成图（双向）

RoPE 如何区分 img_noisy 和 img_cond：
  img_noisy 的 ids: [0, row, col]  → "这是第 0 帧的 (row,col) 位置"
  img_cond  的 ids: [1, row, col]  → "这是第 1 帧的 (row,col) 位置"
  相同 (row,col) 但不同帧的两个 patch，RoPE 赋予它们不同的旋转角度
  → 模型可以感知"这两个 patch 空间位置对应，但来自不同帧"
```

> 💡 **这才是 Kontext 能做 consistent 图像编辑的原因**：条件图的每个 patch 和生成图对应位置的 patch 可以直接通过 Attention 交互，空间对应关系由 RoPE 编码保留。无需任何结构改动，也不需要额外的 cross-attention 层。

#### 与 channel concat 的根本区别

| | Channel Concat | Sequence Concat |
|--|---|---|
| 条件信息粒度 | per-patch 局部特征混合 | 全局——任意 patch 可 attend 到任意条件 patch |
| 序列长度 | 不变 | 增加（N_gen + N_cond） |
| 空间对应 | 隐式（同位置 channel 对齐） | 显式（RoPE 编码空间坐标） |
| Attention 结构 | 不变 | 不变（Self-Attention，只是序列更长）|
| 计算量 | 不增加（仅 img_in 变） | 增加（Attention 是 O(N²)，N 增大）|
| 适合任务 | 结构控制（边缘、深度、mask） | 内容/风格参考（参考图编辑）|

---

### 11.4 方式三：txt 侧 concat（Redux）

#### 核心思路
不把条件图当成 img tokens，而是把它的**视觉特征**（用 SigLIP 等 image encoder 提取）拼接到 `txt tokens` 序列末尾，走文本流的路径影响生成。

#### 源码：`prepare_redux()`

```python
img_cond = encoder(img_cond)     # SigLIP → [B, N_visual, D_visual]
img_cond = img_cond.to(torch.bfloat16)

# ⭐ 直接 cat 到 T5 文本 tokens 后面
txt = t5(prompt)                 # [B, N_text, 4096]
txt = torch.cat((txt, img_cond.to(txt)), dim=-2)
# txt: [B, N_text + N_visual, 4096]

txt_ids = torch.zeros(bs, txt.shape[1], 3)  # txt_ids 全 0（无空间位置）
# ← 视觉 tokens 也用全 0 的 txt_ids，不区分，当作"额外的文本条件"处理
```

#### 对 Attention 的影响

```
Redux 方案：修改的是 txt 序列长度，不是 img 序列

双流块的 txt 流处理长度从 N_text 变为 N_text + N_visual
图像流通过联合 Attention attend 到这些视觉 token
  → 条件图的视觉信息通过"文本通道"流入图像生成

本质：把参考图像当成"图像描述的 token"来用
优点：不增加 img 序列长度，不改变空间分辨率
缺点：丢失了空间位置信息（txt_ids 全 0）
```

---

### 11.5 `denoise()` 完整流程串联

```python
def denoise(model, img, img_ids, txt, txt_ids, vec, timesteps,
            guidance,
            img_cond=None,        # channel concat 用（Fill/Canny）
            img_cond_seq=None,    # sequence concat 用（Kontext）
            img_cond_seq_ids=None):

    guidance_vec = torch.full((img.shape[0],), guidance, ...)

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, ...)

        # ── Step 1: 构建实际输入 ─────────────────────────────────
        img_input    = img
        img_input_ids = img_ids

        if img_cond is not None:                         # 方式一：通道拼接
            img_input = torch.cat((img, img_cond), dim=-1)
            # dim=-1 是特征维，序列长度不变

        if img_cond_seq is not None:                     # 方式二：序列拼接
            img_input    = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
            # dim=1 是序列维，特征维不变

        # ── Step 2: 模型前向 ─────────────────────────────────────
        pred = model(img=img_input, img_ids=img_input_ids,
                     txt=txt, txt_ids=txt_ids, y=vec,
                     timesteps=t_vec, guidance=guidance_vec)

        # ── Step 3: 取回生成图的预测速度 ────────────────────────
        if img_cond_seq is not None:
            pred = pred[:, :img.shape[1]]  # 丢掉条件图 tokens 的输出
            # 条件图本身不参与 denoising，输出无意义

        # ── Step 4: Euler 步更新 ────────────────────────────────
        img = img + (t_prev - t_curr) * pred

    return img
```

---

### 11.6 源码层面的追问 Q&A

**Q: Kontext 的条件图 token 也会被模型"去噪"吗？**
> 不会。条件图 token 作为 `img_cond_seq` 拼入序列后，模型输出 `pred` 的长度是 `N_gen + N_cond`，但 `denoise()` 里 `pred = pred[:, :img.shape[1]]` 直接截断，只取前 `N_gen` 个 token 的预测速度，条件图对应的输出被丢弃。条件图 token 只是"只读的参考信息"。

**Q: Channel concat 为什么需要重训 img_in，而 sequence concat 不需要？**
> Channel concat 改变了每个 token 的特征维度（64 → 128/132），`img_in = nn.Linear(in_channels, hidden_size)` 的 `in_channels` 必须对应修改，所以权重不兼容，必须重新训练或微调。Sequence concat 每个 token 维度不变（仍是 64），只是序列多了几个 token，原有权重完全兼容，可以 zero-shot 泛化（虽然 Kontext 实际上也做了 LADD 蒸馏训练）。

**Q: Kontext 的条件图用什么 img_ids？为什么第 0 维设为 1？**
> `img_cond_ids[..., 0] = 1`，生成图是 0。第 0 维是 RoPE 的"时间轴"，在 `EmbedND` 里 `axes_dim[0]=16` 给时间轴分配了 16 维的旋转空间。这样同一空间位置 (row, col) 的条件帧 token 和生成帧 token 有不同的旋转编码，模型可以区分"这是参考的"还是"这是要生成的"，同时 row/col 相同保证了空间对应关系的感知。

**Q: Redux 的视觉 token 为什么用 txt_ids 全零，不给空间位置？**
> Redux 的目标是"风格/内容参考"而非像素级对齐。用全零 txt_ids 意味着这些视觉 token 和文本 token 一样"没有空间位置"，作为全局语义条件使用。如果给了空间位置，模型会尝试把条件图的空间结构迁移到生成图，反而不是想要的效果。

---

## 十一B、源码逐层精读（基础模块）

> 源码路径：`src/flux/math.py` · `src/flux/modules/layers.py` · `src/flux/model.py`  
> 建议配合上文架构理解阅读，每段代码后附有关键注释

---

### 11.1 `math.py` — RoPE 核心实现

```python
# ── rope()：为某一维度的位置序列生成旋转矩阵 ──────────────────────────
def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    # 生成频率：dim/2 个不同频率的正弦基，越靠后频率越低
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)                  # shape: [dim/2]

    # 外积：每个位置 × 每个频率
    out = torch.einsum("...n,d->...nd", pos, omega) # shape: [..., seq, dim/2]

    # 构造 2×2 旋转矩阵的四个元素：cos, -sin, sin, cos
    out = torch.stack([torch.cos(out), -torch.sin(out),
                       torch.sin(out),  torch.cos(out)], dim=-1)
    # reshape 成 [..., seq, dim/2, 2, 2] 的旋转矩阵形式
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()

# ── apply_rope()：把旋转矩阵作用到 Q 和 K 上 ─────────────────────────
def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    # reshape 成 [..., head_dim/2, 1, 2]，方便矩阵乘法
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    # 旋转 = 复数乘法：[cos, -sin; sin, cos] × [x, y]^T
    # freqs_cis[..., 0] 是旋转矩阵第一行，freqs_cis[..., 1] 是第二行
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), \
           xk_out.reshape(*xk.shape).type_as(xk)

# ── attention()：带 RoPE 的标准 SDPA ─────────────────────────────────
def attention(q, k, v, pe):
    q, k = apply_rope(q, k, pe)              # 先旋转再算注意力
    x = F.scaled_dot_product_attention(q, k, v)  # Flash Attention
    x = rearrange(x, "B H L D -> B L (H D)")     # 合并多头
    return x
```

**关键理解**：
- `rope()` 只生成旋转矩阵，不直接改变 token——旋转在 `apply_rope()` 里发生
- 双流块中图像和文本的 `ids` 先 cat 再统一算 `pe`，所以图文位置编码共享同一套 θ
- FLUX 用 3 轴 ids：`[batch_id, height_id, width_id]`（见 `EmbedND`），文本的 height/width 恒为 0

---

### 11.2 `layers.py` — 模块拆解

#### EmbedND：多维 RoPE 位置编码

```python
class EmbedND(nn.Module):
    def __init__(self, dim, theta, axes_dim: list[int]):
        # axes_dim 例如 [16, 56, 56]，三轴分配的维度之和 = head_dim
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        # ids shape: [B, seq_len, 3]  (3轴: t/h/w)
        # 对每个轴单独算 rope，然后 cat 起来
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)  # 加 head 维度，供多头注意力广播
```

> 💡 `axes_dim=[16, 56, 56]`：head_dim=128 时，16 维给时间轴（txt 用），56+56 给 H/W 轴  
> 文本 token 的 `txt_ids` 全为 0，所以文本不带空间位置——只有图像 patch 有 H/W 位置

---

#### QKNorm + RMSNorm：稳定大模型训练

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        # RMS = sqrt(mean(x^2))，比 LayerNorm 少减均值，更快
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(x.dtype) * self.scale  # 可学习缩放

class QKNorm(nn.Module):
    def __init__(self, dim):
        self.query_norm = RMSNorm(dim)  # 每个 head 独立 norm（dim = head_dim）
        self.key_norm   = RMSNorm(dim)

    def forward(self, q, k, v):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)  # 对齐 dtype（v 是 bfloat16）
```

> 💡 为什么只 norm Q 和 K，不 norm V？  
> 注意力权重 = softmax(QK^T/√d)，Q 和 K 的尺度决定权重分布；V 只是加权求和，尺度异常影响有限

---

#### Modulation：替代 AdaLN 的条件调制

```python
class Modulation(nn.Module):
    def __init__(self, dim, double: bool):
        # double=True（双流块）：输出 6 份（图/文各 shift+scale+gate）
        # double=False（单流块）：输出 3 份（shift+scale+gate）
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec):
        # vec: [B, D]，经过 SiLU 激活后线性映射
        out = self.lin(F.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        # 返回 ModulationOut(shift, scale, gate)，double 时返回两个
        return ModulationOut(*out[:3]), ModulationOut(*out[3:]) if self.is_double else None
```

调制应用方式（对比 AdaLN）：

| 方式 | 公式 | 特点 |
|------|------|------|
| AdaLN | `γ·LN(x) + β` | 经典，无门控 |
| FLUX Modulation | `(1 + scale)·LN(x) + shift`，残差乘以 `gate` | 有 gate，可动态关闭某层 |

```python
# 实际使用（双流块中）：
img_modulated = (1 + img_mod1.scale) * self.img_norm1(img) + img_mod1.shift
img = img + img_mod1.gate * self.img_attn.proj(img_attn)  # gate 控制残差强度
```

---

#### DoubleStreamBlock：双流块完整前向

```python
def forward(self, img, txt, vec, pe):
    img_mod1, img_mod2 = self.img_mod(vec)   # 图像流：2个调制（attn + mlp）
    txt_mod1, txt_mod2 = self.txt_mod(vec)   # 文本流：2个调制（attn + mlp）

    # ── Step 1: 各自准备 QKV ─────────────────────────────────
    img_modulated = (1 + img_mod1.scale) * self.img_norm1(img) + img_mod1.shift
    img_q, img_k, img_v = split(self.img_attn.qkv(img_modulated))
    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)  # QK-Norm

    txt_modulated = (1 + txt_mod1.scale) * self.txt_norm1(txt) + txt_mod1.shift
    txt_q, txt_k, txt_v = split(self.txt_attn.qkv(txt_modulated))
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)  # QK-Norm

    # ── Step 2: 图文 QKV 拼接，统一算注意力 ───────────────────
    # ⭐ 核心：虽然权重独立，但注意力是联合计算的！
    q = torch.cat((txt_q, img_q), dim=2)   # [B, H, N_txt+N_img, D]
    k = torch.cat((txt_k, img_k), dim=2)
    v = torch.cat((txt_v, img_v), dim=2)
    attn = attention(q, k, v, pe=pe)       # pe 同时含 txt+img 的位置信息

    # ── Step 3: 切分回各自的流，更新残差 ──────────────────────
    txt_attn = attn[:, :txt.shape[1]]
    img_attn = attn[:, txt.shape[1]:]

    img = img + img_mod1.gate * self.img_attn.proj(img_attn)
    img = img + img_mod2.gate * self.img_mlp(
        (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
    )
    txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
    txt = txt + txt_mod2.gate * self.txt_mlp(...)

    return img, txt
```

> 💡 **精华所在**：图文 Q/K/V 来自各自独立的线性层（不共享权重），但 Attention 矩阵是联合计算的  
> 这意味着：图像 token 的 Q 可以 attend 到文本 token 的 K，实现隐式 Cross-Attention——**比显式 Cross-Attn 更高效**

---

#### SingleStreamBlock：并行注意力的实现

```python
def forward(self, x, vec, pe):
    mod, _ = self.modulation(vec)   # 只有 1 个调制（单流）
    x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift

    # ⭐ 并行的关键：一个 linear1 同时输出 QKV 和 MLP 的输入
    # linear1: hidden -> 3*hidden + mlp_hidden
    qkv, mlp = torch.split(
        self.linear1(x_mod),
        [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
    )

    q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    q, k = self.norm(q, k, v)

    attn = attention(q, k, v, pe=pe)     # 注意力路径
    # mlp 路径：直接 GELU 激活（已经是中间激活值了）

    # linear2: (hidden + mlp_hidden) -> hidden，两路结果在这里合并
    output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=2))
    return x + mod.gate * output
```

**并行注意力（Parallel Attention）图解**：

```
x_mod ─→ linear1 ─┬─→ QKV ─→ Attention ─→ attn    ─┐
                   └─→ mlp ─→    GELU   ─→ mlp_act ─┤
                                                      └─→ cat ─→ linear2 ─→ output
```

> 💡 对比串行（标准 Transformer）：  
> 串行：`x → Attn → x' → FFN → x''`（两次完整前向）  
> 并行：Attn 和 FFN 共用同一次 Norm 和第一个 Linear，**节省约 1/3 计算量**，是 GPT-J 的思路

---

### 11.3 `model.py` — 完整前向流程

```python
class Flux(nn.Module):
    def forward(self, img, img_ids, txt, txt_ids, timesteps, y, guidance=None):
        # ── 1. 输入 Embedding ──────────────────────────────────────
        img = self.img_in(img)        # [B, N_patch, 64] → [B, N_patch, hidden]
        txt = self.txt_in(txt)        # [B, N_txt, 4096] → [B, N_txt, hidden]
                                      # T5 输出 4096 维，投影到 hidden_size

        # ── 2. 构造全局条件向量 vec ────────────────────────────────
        vec = self.time_in(timestep_embedding(timesteps, 256))
        #     ↑ 时间步 t → sinusoidal 256d → MLP → hidden_size

        if self.params.guidance_embed:            # dev 模型才有
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
            # ↑ guidance scale → sinusoidal 256d → MLP → 加到 vec
            # schnell 没有 guidance_embed，所以这里是 nn.Identity()（不加）

        vec = vec + self.vector_in(y)
        #           ↑ y 是 CLIP pooled embedding → MLP → 加到 vec
        # 最终 vec = f(t) + f(guidance) + f(CLIP)，调制所有 block

        # ── 3. 位置编码 ────────────────────────────────────────────
        ids = torch.cat((txt_ids, img_ids), dim=1)  # 图文 ids 拼接
        pe  = self.pe_embedder(ids)                  # 统一算 RoPE
        # txt_ids 全为 0（无空间位置），img_ids 含 [0, row, col]

        # ── 4. 双流块 ──────────────────────────────────────────────
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
            # img 和 txt 各自更新，但注意力联合计算

        # ── 5. 单流块 ──────────────────────────────────────────────
        img = torch.cat((txt, img), dim=1)  # ⭐ 文本拼到前面！
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1]:, ...]   # ⭐ 去掉文本部分，只保留图像

        # ── 6. 输出 ────────────────────────────────────────────────
        img = self.final_layer(img, vec)
        # AdaLN 调制 + Linear → [B, N_patch, patch_size² × out_channels]
        # 后续 unpatchify → [B, 16, H/2, W/2]（再经 VAE decoder）
        return img
```

**FluxParams 对应 FLUX.1-dev 的实际配置**：

```python
# FLUX.1 [dev / schnell] 实际参数
FluxParams(
    in_channels        = 64,    # 16ch VAE × 2×2 patch = 64
    out_channels       = 64,
    vec_in_dim         = 768,   # CLIP ViT-L pooled dim
    context_in_dim     = 4096,  # T5-XXL hidden dim
    hidden_size        = 3072,  # Transformer hidden dim
    mlp_ratio          = 4.0,
    num_heads          = 24,    # head_dim = 3072/24 = 128
    depth              = 19,    # 19 个双流块
    depth_single_blocks= 38,    # 38 个单流块（单流是双流的 2 倍！）
    axes_dim           = [16, 56, 56],   # RoPE 三轴维度，合计 128 = head_dim
    theta              = 10000,
    qkv_bias           = True,
    guidance_embed     = True,  # dev=True, schnell=False
)
```

> 💡 **38 个单流 vs 19 个双流**：单流块参数少（权重共享），堆更多层来补偿表达能力，是性价比权衡

---

### 11.4 `sampling.py` — 数据预处理（prepare 函数）

```python
def prepare(t5, clip, img, prompt):
    bs, c, h, w = img.shape  # latent 尺寸，如 [1, 16, 128, 128]

    # ── Patchify：2×2 patch 化 ──────────────────────────────────
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    # [B, 16, 128, 128] → [B, 64×64, 64]  (h=128/2=64, 64×16×4=64维)

    # ── 图像位置 id：记录每个 patch 的 (0, row, col) ────────────
    img_ids = torch.zeros(h//2, w//2, 3)
    img_ids[..., 1] = torch.arange(h//2)[:, None]  # row 坐标
    img_ids[..., 2] = torch.arange(w//2)[None, :]  # col 坐标
    # img_ids[..., 0] = 0，保留给多图/视频场景（Kontext 中会用到）

    # ── 文本编码 ─────────────────────────────────────────────────
    txt = t5(prompt)           # T5 sequence tokens: [B, seq_len, 4096]
    txt_ids = torch.zeros(bs, txt.shape[1], 3)  # 文本无空间位置，全 0
    vec = clip(prompt)         # CLIP pooled: [B, 768]

    return {"img": img, "img_ids": img_ids, "txt": txt,
            "txt_ids": txt_ids, "vec": vec}
```

> 💡 `img_ids` 第 0 维（时间轴）全为 0，这是为视频/多图扩展预留的设计：  
> 在 FLUX.1 Kontext 中，把多帧图像的 img_ids 的第 0 维设为帧编号，就实现了时序感知

---

### 11.5 源码层面的常见面试追问

**Q: 双流块中图文注意力是怎么交互的，是 Cross-Attention 吗？**
> 不是传统 Cross-Attention。图文各有独立的 QKV 线性层（独立权重），但在计算 Attention 时 Q/K/V 被 cat 在一起统一做 Self-Attention。这样图像 Q 能 attend 到文本 K，文本 Q 也能 attend 到图像 K，效果等价于双向 Cross-Attention，但只需一次矩阵乘法，更高效。

**Q: guidance 信息是怎么传入模型的？**
> `guidance_scale` → `timestep_embedding(guidance, 256)` 转成 sinusoidal 嵌入 → MLP → 加到 `vec`。`vec` 再通过每个 block 的 `Modulation` 层影响 shift/scale/gate。所以 guidance 信号**渗透到每一层的每一个 token**，而不只是输入层。

**Q: 单流块进入时为什么把 txt 放在 img 前面？**
> `img = torch.cat((txt, img), dim=1)`，文本在前。这只是约定，出来后 `img = img[:, txt.shape[1]:]` 把文本部分丢掉。文本放前面的好处是 attention mask 如果有的话更好切片，且与双流块输出时 txt/img 的顺序保持一致（双流块 `attn[:, :txt.shape[1]]` 切文本）。

**Q: 为什么 schnell 的 guidance_embed=False，能做到 guidance_scale=0？**
> schnell 的 `FluxParams.guidance_embed=False`，所以 `self.guidance_in = nn.Identity()`，guidance 路径直接跳过，vec 里根本没有 guidance 信号。这不是 guidance_scale 设为 0，而是模型压根没有 guidance 嵌入层——guidance distillation 的效果已经在蒸馏训练时"内化"进权重了。


---

## 十二、面试高频问题 & 答案思路

> 以下问题来自两个维度：**概念层**（理解原理）和**源码层**（代码细节），后者是加分项

**Q1: FLUX 的双流和单流分别做什么，为什么要这样设计？**
> 双流在前：图文各自提取特征，通过联合 Attention 相互影响，各保留模态独立性。单流在后：concat 后联合处理，实现深度融合。这种设计兼顾了"各自充分建模 + 深度融合"。源码中双流共 19 层，单流 38 层，单流更多因为权重参数少（共享）。

**Q2: FLUX 双流块的图文交互是 Cross-Attention 吗？**
> 不是传统 Cross-Attention。图文各有独立 QKV 线性层，但 Attention 计算时 Q/K/V 被 cat 在一起统一做 Self-Attention（`q = cat(txt_q, img_q)`）。效果等价于双向 Cross-Attention，但只需一次矩阵乘，更高效。

**Q3: FLUX 和 SD3 的 MMDiT 有什么区别？**
> FLUX 在 SD3 双流基础上增加了：① 大量 Single Stream Blocks（SD3 基本全是双流）；② QK-Norm（RMSNorm on Q,K）和并行注意力；③ Guidance Distillation；④ 规模扩到 12B；⑤ RoPE 三轴位置编码（新增时间轴）。

**Q4: Flow Matching 和 DDPM 的核心区别？**
> Flow Matching 学速度场，轨迹是直线，训练目标 $v = \epsilon - x_0$；DDPM 学噪声 $\epsilon$，轨迹是弯曲的马尔可夫链。FM 采样步数更少，概念更简洁。

**Q5: dev 和 schnell 的蒸馏有什么本质区别？**
> dev 是 Guidance Distillation（单次前向替代 CFG 两次前向，步数不变仍需 50 步）；schnell 是在此基础上再加 Timestep Distillation+LADD（4 步即可）。源码体现：dev 的 `guidance_embed=True` 有 `guidance_in` MLP 层，schnell 是 `nn.Identity()`。

**Q6: guidance scale 是怎么传入模型的？**
> `guidance` → `timestep_embedding(guidance, 256)` → MLP(`guidance_in`) → 加到 `vec`。`vec` 再经每层的 `Modulation` 预测 shift/scale/gate，作用于每个 token。所以 guidance 信号渗透到所有层的所有 token，不只是输入层。

**Q7: 为什么 FLUX 要同时用 T5 和 CLIP？**
> CLIP 擅长图文对齐，输出 pooled embedding 加入 `vec` 做全局调制；T5 理解复杂长文本，输出 sequence tokens 参与双流块的联合 Attention。两者角色完全不同，互补而非冗余。

**Q8: RoPE 相比传统位置编码的优势？**
> RoPE 编码相对位置，支持任意分辨率外推；FLUX 用 3 轴 RoPE（时间/高/宽），其中 `axes_dim=[16,56,56]`，文本 token 的空间轴为 0（无空间位置），天然区分图文。

---

```
Stable Diffusion 1.x (U-Net + CLIP + 4ch VAE)
        ↓ 改了主干
SD3 (MMDiT 双流 + T5/CLIP + 16ch VAE + FM)
        ↓ 扩大规模 + 增加单流 + QK-Norm + 蒸馏
FLUX.1 (12B MMDiT + 双流/单流 + RoPE + 2种蒸馏)
        ↓ 未来方向
视频生成（CogVideoX/Wan/HunyuanVideo 都借鉴了类似思路）
```

---

## 十三、推荐资源

- **官方博客**: [Black Forest Labs 公告](https://bfl.ai/announcing-black-forest-labs/)
- **Hugging Face Model Card**: [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- **代码仓库**: [black-forest-labs/flux](https://github.com/black-forest-labs/flux)
- **架构解析（中文）**: 搜索"FLUX架构解析 zzfive CSDN"
- **Flow Matching 原理**: [Flow Matching for Generative Modeling (Lipman et al., 2022)](https://arxiv.org/abs/2210.02747)
- **RoPE 原理**: [RoFormer (Su et al., 2021)](https://arxiv.org/abs/2104.09864)

---

## 一个思考题（面试前做一下）

> 假设你要让 FLUX.1 支持视频生成（每帧之间需要时序一致性），你会在哪些地方做修改？提示：想想 RoPE 的维度、单流双流的 token 构成、以及 VAE 的改动。

（参考答案方向：3D RoPE + 时序 token 拼接 + 3D VAE，这正是 CogVideoX / HunyuanVideo 的思路）
