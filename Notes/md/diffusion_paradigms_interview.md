# 扩散模型三大范式：DDPM / DDIM / Flow Matching

> 面试速查笔记 · 2026/04/11
> 已有基础：了解 DDPM、DDIM、Flow Matching 各自的大致概念，但不清楚三者的内在联系与区别

---

## 🗺️ 知识地图

```
[概率扩散 DDPM ✓]
       ↓ 加速采样（确定性化）
[DDIM ✓] ← 去掉随机性，DDPM 的特例
       ↓ 换一套框架（直线路径 + 速度场）
[Flow Matching ✓] ← 更通用、更高效的生成框架

三者本质：都是在学习"把噪声变成数据"的路径
区别：路径形状不同 / 训练目标不同 / 采样效率不同
```

---

## 一、DDPM：扩散模型的奠基

### 直觉

> **往图片里一点点加噪声，训练模型学会反向一步步去噪。**

### 正向过程（加噪）

将数据 $x_0$ 逐步加噪，共 $T$ 步（通常 $T=1000$）：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t;\, \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t \mathbf{I})$$

利用重参数化技巧，可以**一步到位**采样任意时刻的噪声图：

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

其中 $\bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s)$。

### 训练目标

训练网络 $\epsilon_\theta$ 预测加入的噪声：

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

### 反向采样（去噪）

逐步去噪，**每步都要加随机噪声**（马尔可夫链）：

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right) + \sigma_t z, \quad z \sim \mathcal{N}(0,\mathbf{I})$$

### 核心特点

| 特征 | 说明 |
|------|------|
| 路径形状 | 随机游走（非直线），轨迹弯曲 |
| 采样步数 | 需要 **1000 步**，很慢 |
| 随机性 | 每步都注入噪声，采样是随机的 |
| 理论基础 | 非平衡热力学 / 随机微分方程（SDE） |
| 训练目标 | 预测噪声 $\epsilon$ |

### 常见误区

- ❌ "DDPM 每步只能去噪一点" → ✅ 正向加噪可以一步跳到任意 $t$，但**反向采样**确实是逐步的
- ❌ "T 越大越好" → ✅ T 大精度高但采样慢，需要权衡

---

## 二、DDIM：DDPM 的确定性加速版

### 直觉

> **DDPM 的反向过程是随机的（每步加噪声），DDIM 把这个随机性去掉，换成确定性 ODE，从而可以大步跳跃。**

### 关键推导

DDPM 的反向过程本质上对应一个 SDE。DDIM 找到了它对应的**确定性 ODE**（概率流 ODE）：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{预测的 }x_0} + \underbrace{\sqrt{1-\bar{\alpha}_{t-1}}\,\epsilon_\theta}_{\text{方向项}}$$

注意：**这里没有随机噪声项 $z$**，完全确定！

### DDIM 和 DDPM 的关系

```
DDPM：SDE（有随机噪声）
  ↓ 令随机项系数 → 0
DDIM：ODE（确定性轨迹）
```

- **DDIM 是 DDPM 的特例**（$\eta=0$ 时），两者用**同一个训练好的模型**，只是采样方式不同
- $\eta \in [0,1]$ 控制随机性：$\eta=0$ 是纯 DDIM，$\eta=1$ 退化回 DDPM

### 为什么可以跳步？

因为 ODE 是确定性的，轨迹是平滑曲线，可以用大步长近似积分（DDPM 的随机性让大步长误差大）。

$$\text{DDPM: 1000步} \xrightarrow{\text{DDIM}} \text{20\~{}50步即可}$$

### 核心特点

| 特征 | 说明 |
|------|------|
| 路径形状 | 比 DDPM 更平滑，但仍非直线 |
| 采样步数 | **20~50 步**，速度大幅提升 |
| 随机性 | 默认**确定性**（同一噪声 → 同一输出） |
| 理论基础 | ODE / 概率流 |
| 训练目标 | 与 DDPM **完全相同**（预测 $\epsilon$），不需要重新训练！ |

### 常见误区

- ❌ "DDIM 需要重新训练模型" → ✅ **不需要**，直接换采样器就行
- ❌ "DDIM 比 DDPM 质量差" → ✅ 步数够多时质量相当，甚至某些情况更好（因为消除了随机误差）

---

## 三、Flow Matching：更优雅的新范式

### 直觉

> **不走弯路，直接走直线！Flow Matching 用线性插值定义"噪声 → 数据"的最优传输路径，训练模型学习这条直线的速度场。**

### 符号约定（注意与 DDPM 方向相反！）

- $x_0$：**真实数据**（干净图片），$t=0$
- $x_1$：**纯噪声**，$t=1$
- 推理：从 $x_1$（噪声）积分走到 $x_0$（图片）

### 正向路径（Rectified Flow）

任意时刻的"混合样本"定义为**线性插值**：

$$x_t = (1-t)\,x_0 + t\,x_1, \quad t \in [0, 1]$$

**这就是直线！** 相比 DDPM 的弯曲随机游走路径，Flow Matching 的路径是最短的。

### 训练目标：回归速度场

训练模型 $v_\theta(x_t, t)$ 拟合每个位置的"速度"（即路径切线方向）：

$$\mathcal{L}(\theta) = \mathbb{E}_{t,\, x_0,\, x_1}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$$

目标速度 $x_1 - x_0$ 是**常数**——方向就是从数据指向噪声，一直不变。

### 推理：ODE 积分

从 $x_1$ 出发，跟着速度场走：

$$\frac{dx_t}{dt} = v_\theta(x_t, t) \implies x_{t-\Delta t} = x_t - v_\theta(x_t, t) \cdot \Delta t$$

### 核心特点

| 特征 | 说明 |
|------|------|
| 路径形状 | **直线**（最优传输），最平滑 |
| 采样步数 | 理论上**更少步**（~10步），实际同类质量所需步数最少 |
| 随机性 | 默认**确定性 ODE** |
| 理论基础 | 最优传输 / 连续归一化流（CNF） |
| 训练目标 | 预测**速度** $v = x_1 - x_0$，而非噪声 $\epsilon$ |
| 代表模型 | **SD3, FLUX, Stable Video Diffusion** |

### 常见误区

- ❌ "Flow Matching 和 DDIM 是一回事" → ✅ 框架不同：DDIM 是 DDPM 的采样方法，FM 是独立的训练范式
- ❌ "Flow Matching 的路径是真正的直线" → ✅ 每对 $(x_0, x_1)$ 单独是直线，但混合后的**边际速度场是弯的**；只有学到完美模型时才近似直线

### 💡 深挖：Flow Matching 路径是直线，为什么还需要多步采样？

这是一个非常好的问题，也是面试里容易被追问的点。

#### 理想情况 vs 现实情况

**理想情况（训练完美）：**
每对 $(x_0, x_1)$ 之间的路径确实是直线，速度 $v = x_1 - x_0$ 是常数。  
如果模型学得完美，从 $x_1$ 出发沿速度场走，**理论上一步就能到 $x_0$**。

**现实情况（为什么不行）：**

> **根本原因：同一个 $x_t$，来自无数条不同的直线路径——模型只能给出"平均速度"，而平均方向不等于任何一条真实直线。**

具体来说，训练时 $x_t = (1-t)x_0 + tx_1$，其中 $x_0$ 和 $x_1$ 是**独立随机采样**的。  
不同的 $(x_0, x_1)$ 对可能经过**同一个 $x_t$ 位置**，但目标速度方向完全不同：

```
x_1_a ────→ x_t ────→ x_0_a    （路径 A，速度向左下）
x_1_b ────→ x_t ────→ x_0_b    （路径 B，速度向右上）
              ↑
        同一个点 x_t
```

模型在 $x_t$ 处学到的是**所有经过这里的路径的期望速度**：

$$v_\theta(x_t, t) \approx \mathbb{E}[x_1 - x_0 \mid x_t]$$

这个**期望方向**不等于任何一条真实直线的方向，所以如果真的一步走过去，会走偏。

#### 类比理解

想象你在城市里问路，路过的人各自要去不同的目的地。你问"请问往哪走？"，大家的平均指向是**城市中心**，但没有人真的住在城市中心——沿着平均方向走一步，你并不会到达任何真实目的地。

需要**走一步再问一次**，不断修正方向，才能最终到达正确的地方。

#### 步数越多，误差越小

$$\text{1步：沿期望速度走全程，偏差最大}$$
$$\text{多步：每步走一小段，中途修正，误差累积更小}$$

步数越多，ODE 积分越精确，生成质量越高。Flow Matching 相比 DDPM 的优势是：**同样步数下，因为路径曲率更小，误差更小**；或者说**要达到同等质量，所需步数更少**。

#### 一句话总结

> **单对路径是直线，但"边际速度场"（学到的期望速度场）是弯的。模型每一步只能走"当前平均方向"，需要多步不断修正，才能到达真实数据分布。**

---

## 四、三者对比总览

| | DDPM | DDIM | Flow Matching |
|---|---|---|---|
| **本质** | SDE（随机过程） | ODE（确定性） | ODE（最优传输） |
| **路径** | 弯曲随机游走 | 弯曲但平滑 | 线性（最短路径） |
| **采样步数** | ~1000 步 | ~20-50 步 | ~10-20 步 |
| **训练目标** | 预测噪声 $\epsilon$ | 同 DDPM（无需重训） | 预测速度 $v = x_1-x_0$ |
| **随机性** | 有（每步加噪） | 可选（$\eta$ 控制） | 无（默认 ODE） |
| **需要重训** | —— | ❌ 不需要 | ✅ 需要重训 |
| **代表作** | DDPM (Ho 2020) | DDIM (Song 2021) | Flow Matching (Lipman 2022) / Rectified Flow (Liu 2022) |
| **当前主流** | 理论基础 | 常用采样器 | **工业界主流**（FLUX, SD3） |

---

## 五、内在联系：统一视角

### 它们都是"生成轨迹"问题

三者本质上都在回答同一个问题：

> **如何定义一条从噪声到数据的路径，并训练模型沿这条路径走？**

```
DDPM: 随机路径（SDE），噪声级别由方差调度 β_t 控制
  │
  └─→ DDIM: 把 SDE 变成 ODE，路径形状不变，去掉随机项
              │
              └─→ Flow Matching: 重新设计路径（直线），更换训练目标
```

### Score Function 视角（更深层联系）

- DDPM 训练的 $\epsilon_\theta$ 实际上在估计**分数函数（score function）** $\nabla_{x_t} \log p(x_t)$：
  $$\epsilon_\theta(x_t, t) \approx -\sqrt{1-\bar{\alpha}_t} \cdot \nabla_{x_t} \log p(x_t)$$

- Flow Matching 的速度场 $v_\theta$ 可以通过以下关系与分数函数联系：
  $$v_\theta \approx \frac{x_1 - x_t}{1-t} \quad \text{（高斯情况下）}$$

- **DDIM 的确定性采样 = 对 score-based SDE 的概率流 ODE 求解**

### 从 DDPM 推出 Flow Matching 的直觉

DDPM 的路径弯曲，是因为每步加的噪声方差 $\beta_t$ 是非线性的，导致路径是非线性的。  
Flow Matching 说：**我直接用线性插值构造路径，反正我只需要路径的切向量（速度），不需要马尔可夫链属性。**

---

## 六、面试高频问题 & 答题要点

### Q1：DDPM 和 DDIM 的区别？

**答：** DDIM 是 DDPM 的加速采样方法，**不需要重新训练模型**。DDPM 对应 SDE（每步需要加随机噪声），DDIM 找到了对应的确定性 ODE，去掉随机项后轨迹变平滑，可以用大步长近似，从 1000 步加速到 20-50 步。两者用同一个网络（预测 $\epsilon$）。

### Q2：Flow Matching 和 DDPM/DDIM 的本质区别？

**答：** 有两点核心区别：
1. **路径设计不同**：DDPM 用非线性方差调度（弯曲路径），Flow Matching 用线性插值（直线路径），路径更短更平滑，理论上需要更少步数。
2. **训练目标不同**：DDPM 预测噪声 $\epsilon$，FM 直接预测速度场 $v = x_1 - x_0$。FM **需要重新训练**，不能复用 DDPM 的权重。

### Q3：为什么现在工业界（FLUX, SD3）用 Flow Matching 而不是 DDPM？

**答：** 主要三个优势：
- **采样更快**：直线路径导致 ODE 更易数值积分，步数更少
- **训练更稳定**：目标速度是常数方向 $x_1-x_0$，信噪比均匀；DDPM 不同 $t$ 的损失量级差异大
- **更容易扩展到视频/高分辨率**：更少的步数意味着更低的显存消耗

### Q4：Flow Matching 的训练目标为什么是 $x_1 - x_0$？

**答：** 因为路径是线性插值 $x_t = (1-t)x_0 + tx_1$，对 $t$ 求导得到：
$$\frac{dx_t}{dt} = x_1 - x_0$$
这就是路径切线方向，即速度。训练目标就是让网络预测这个常数速度向量。

### Q5：DDIM 可以有随机性吗？

**答：** 可以。DDIM 论文引入了参数 $\eta \in [0,1]$：$\eta=0$ 是确定性 ODE（标准 DDIM），$\eta=1$ 退化回 DDPM 的随机采样。通过调节 $\eta$ 可以在确定性和多样性之间权衡。

---

## 七、关键公式速查

### DDPM 核心

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon \quad \text{（正向，一步到任意t）}$$
$$\mathcal{L} = \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \quad \text{（训练目标）}$$

### DDIM 采样

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_\theta(x_t, t), \quad \hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$$

### Flow Matching 核心

$$x_t = (1-t)x_0 + tx_1 \quad \text{（线性插值路径）}$$
$$\mathcal{L} = \|v_\theta(x_t, t) - (x_1 - x_0)\|^2 \quad \text{（训练目标）}$$
$$x_{t-\Delta t} = x_t - v_\theta(x_t, t) \cdot \Delta t \quad \text{（ODE 采样）}$$

---

## 八、进阶方向

```
DDPM / DDIM / Flow Matching
       ↓
├── Consistency Models（蒸馏 DDPM，1步生成）
├── Stable Diffusion 3 / FLUX（Flow Matching 的工业实现）
├── Stochastic Interpolants（统一 DDPM 和 FM 的理论框架）
└── Score Distillation（SDS，用于 NeRF/3D 生成）
```

### 推荐资源

| 资源 | 说明 |
|------|------|
| [DDPM 论文](https://arxiv.org/abs/2006.11239) | Ho et al. 2020，奠基之作 |
| [DDIM 论文](https://arxiv.org/abs/2010.02502) | Song et al. 2021，必读 |
| [Flow Matching 论文](https://arxiv.org/abs/2210.02747) | Lipman et al. 2022 |
| [Rectified Flow](https://arxiv.org/abs/2209.03003) | Liu et al. 2022，SD3/FLUX 的基础 |
| [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | Lilian Weng 博客，直觉最好 |

### 一个思考题

> DDIM 是把 DDPM 的 SDE 变成 ODE，Flow Matching 也是 ODE——那么 **DDIM 采样和 Flow Matching 采样有什么本质不同**？
>
> 提示：思考两者的**路径曲率**（是否是直线？）以及**是否可以用同一个网络**。

---

*笔记整理于 2026/04/11，基于已有 `flow_matching_basics.tex` 内容扩展*
