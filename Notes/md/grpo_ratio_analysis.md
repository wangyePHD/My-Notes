# GRPO Importance Ratio 深度排查笔记

> 记录时间：2025-04-11  
> 背景：R3（BAGEL 图像编辑 RL 训练）中关于 GRPO importance ratio 的实现方式引发的疑问，经过查阅源码、GitHub 讨论和论文，最终得出结论。

---

## 一、起点：疑问

在 `train/online_rl.py` 中，image policy loss 的写法是：

```python
img_per_token_loss = -(
    torch.exp(log_prob_per_image - log_prob_per_image.detach()) * cur_image_advantages
).sum() / (1e-4 + total_image_num)
```

文本 policy loss 也类似：

```python
per_token_loss = -(
    torch.exp(per_token_logps - per_token_logps.detach()) * cur_text_advantages
).sum() / (1e-4 + total_text_grad_tokens)
```

**疑问**：`log_prob - log_prob.detach()` 永远等于 0，`exp(0) = 1`，那 ratio 不就永远是 1 吗？`.detach()` 怎么能作为 old policy？

---

## 二、对比参考：flow_grpo 的实现

来源：https://github.com/yifan123/flow_grpo

flow_grpo 的做法：

```python
# 阶段 1：rollout 时保存 old log_prob（权重为 θ₀）
images, latents, log_probs = pipeline_with_logprob(...)
samples["log_probs"] = log_probs   # ← 存入内存，这才是真正的 old policy

# 阶段 2：多次梯度更新（θ 从 θ₀ 变成 θ₁, θ₂, ...）
for epoch in range(num_epochs):
    log_prob_new = recompute_log_prob(latents)   # 当前权重重新算
    ratio = torch.exp(log_prob_new - sample["log_probs"])  # π_new / π_old，真的不等于 1
    
    unclipped = -advantages * ratio
    clipped   = -advantages * torch.clamp(ratio, 1-ε, 1+ε)
    loss = torch.mean(torch.maximum(unclipped, clipped))
```

flow_grpo 的 old policy = 真正 rollout 时存下来的 θ₀ log_prob，多次更新后 θ 漂移，ratio 真的偏离 1，clip 有实际意义。

---

## 三、社区讨论的结论

### 来源 1：TRL PR #2565

https://github.com/huggingface/trl/pull/2565#issuecomment-2595837761

TRL 作者 `qgallouedec` 的回答（原文）：

> "The math is correct; if you look at the loss, it is indeed equal to the KL value. However, in terms of differentiation, we cannot remove the ratio term justifying that it equals 1."

他指出，必须保留 ratio 项的形式：

$$\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} = \frac{\pi_\theta}{\left[\pi_\theta\right]_{\cancel{\nabla}}}$$

写成 `exp(logp - logp.detach())` 的原因：**不是为了模拟 old policy，而是为了保留梯度通道**。

如果直接删掉 ratio，loss 变成常数（`-advantage`），梯度为 0，模型不更新。

### 来源 2：R1-V Issue #174

https://github.com/StarsfieldAI/R1-V/issues/174

有人认为这是 bug（"open-r1 库一开始犯了这个错误，现在已纠正"），有人认为不是。

**结论**：两种都是正确实现，只是适用场景不同：

| | `logp.detach()` 方式（TRL/R3） | 存储真实 old logp 方式（flow_grpo/open-r1 修正版）|
|---|---|---|
| 每次 rollout 做几次梯度更新 | **1 次** | **多次**（num_epochs）|
| old policy 来源 | 当前 forward 的 stop-grad 拷贝 | rollout 时真正存下来的 θ₀ |
| ratio 数值 | 永远 = 1 | 真的偏离 1 |
| clip 有用吗 | 无用（形同虚设） | 有用 |
| 数学自洽吗 | ✅ 单步更新下自洽 | ✅ 多步更新下自洽 |

### 来源 3：TRL Issue #2608（最终结论）

https://github.com/huggingface/trl/issues/2608

`qgallouedec` 明确说明（Comment 3，原文）：

> "In the current implementation, we're just update once after a generation. In fact, we align with this sentence from the paper: *The policy model only has a single update following each exploration stage.*
> Therefore it implies that π_θ_old = π_θ, and the equation can be simplified to..."

$$\mathcal{J}_{\text{GRPO}}(\theta) = \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \frac{\pi_\theta}{\left[\pi_\theta\right]_{\cancel{\nabla}}} \hat{A}_{i,t} - \beta \mathbb{D}_{\text{KL}}\left[\pi_\theta \| \pi_{\text{ref}}\right] \right]$$

用户 `ghrua` 进一步指出（Comment 10，关键）：

> "I think the two lines of code are equivalent... the single-update GRPO degrades to the standard policy gradient, which is consistent with Eq.(17) in DeepSeekMath."

```python
# 这两行完全等价（梯度相同）
loss = -torch.exp(per_token_logps - per_token_logps.detach()) * advantages
loss = -per_token_logps * advantages   # ← 就是 REINFORCE
```

证明：链式法则下，`∂/∂θ [exp(logp - logp.detach())] = exp(0) × ∂logp/∂θ = ∂logp/∂θ`

---

## 四、最终结论

### R3 的 GRPO loss 本质上是什么？

**带 group-relative advantage 归一化的 REINFORCE（Policy Gradient）**

```
loss = -log_prob × advantage
```

其中 advantage 用 GRPO 的 group-relative 方式归一化（组内减均值除标准差）。

### 三个层次的理解

| 层次 | 结论 |
|---|---|
| **数值上** | `exp(logp - logp.detach()) = 1`，ratio 永远等于 1 |
| **梯度上** | 等价于直接写 `logp × advantage`，即标准 REINFORCE |
| **算法上** | 单步 GRPO = REINFORCE + GRPO 的 group-relative advantage |

### R3 和 flow_grpo 的本质区别

| | R3（单步） | flow_grpo（多步） |
|---|---|---|
| 算法名称 | REINFORCE + group advantage | GRPO with importance sampling |
| 采样效率 | 低（每个 rollout 只用 1 次） | 高（每个 rollout 复用 num_epochs 次）|
| 实现复杂度 | 简单 | 需要存储 old_log_prob，处理多轮更新 |
| 是否需要 clip | 不需要（ratio=1，clip 无效） | 需要（ratio 真的偏离 1）|
| 是否正确 | ✅ 单步下数学自洽 | ✅ 多步下数学自洽 |

### ref_model 的角色（常见混淆点）

`ref_model` **不是** old policy，和 importance ratio 没有关系。

- **old policy**：用于 importance ratio（π_new / π_old），在单步实现中退化为 `logp.detach()`
- **ref_model**：用于 KL penalty（防止训练后的策略偏离预训练模型太远），是一个**冻结的预训练权重快照**

```python
# KL penalty（ref_model 的用途）
kl = exp(ref_logp - logp) - (ref_logp - logp) - 1   # K3 估计

# importance ratio（和 ref_model 无关）
ratio = exp(logp - logp.detach())   # = 1
```

---

## 五、如果想升级为真正的多步 GRPO

R3 中 `RolloutStepResult` 已经存了 `log_prob`，基础设施完备。需要以下改动：

**1. rollout 时标记 old_log_prob**（已有，无需改动）
```python
rollout_result.log_prob = log_probs[i]   # 这就是 old log_prob
```

**2. 训练循环中重新计算 new log_prob**
```python
for epoch in range(num_epochs):   # 新增多次更新
    log_prob_new = recompute_from_packed_input(...)
    old_log_prob = loss_info["old_log_probs"]   # 从 rollout 存的值取
    
    ratio = torch.exp(log_prob_new - old_log_prob)   # 真实 ratio
    
    unclipped = -advantages * ratio
    clipped   = -advantages * torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    loss = torch.mean(torch.maximum(unclipped, clipped))
```

这是**性能优化方向**（更高的采样效率），而不是纠正错误。

---

## 六、补充：什么是 REINFORCE（Policy Gradient）

### 背景

REINFORCE 是 1992 年 Williams 提出的最基础的策略梯度算法，是所有现代 RL（PPO、GRPO 等）的祖先。

### 核心思想

智能体（策略 π_θ）采取行动 a，获得奖励 R。目标是最大化期望奖励：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

对 θ 求梯度，利用 log-derivative trick：

$$\nabla_\theta J(\theta) = \mathbb{E}\left[ R(\tau) \cdot \nabla_\theta \log \pi_\theta(a|s) \right]$$

**直觉**：如果某个行动得到了高奖励，就增大它的概率（`log π` 增大）；如果得到了低奖励，就减小它的概率。

### 对应的 loss 写法（PyTorch 风格）

```python
loss = -(log_prob * reward).mean()
#        ↑ 要最大化 J，等价于最小化 -J
#        ↑ log π_θ(a|s)
#                  ↑ 奖励 R（或 advantage A）
```

**为什么用 log_prob 而不是 prob？**  
直接对 prob 求梯度会有数值不稳定问题，log_prob 的梯度更稳定：

$$\nabla_\theta \pi_\theta = \pi_\theta \cdot \nabla_\theta \log \pi_\theta$$

### Advantage 版本（减 baseline）

原始 REINFORCE 方差很大。用 advantage `A = R - b`（b 是 baseline，通常是 value function 的预测值）代替原始奖励，可以大幅降低方差：

```python
loss = -(log_prob * advantage).mean()
```

**直觉**：
- `A > 0`：这个行动比平均水平好 → 增大概率
- `A < 0`：这个行动比平均水平差 → 减小概率
- `A = 0`：和平均水平一样 → 不更新

### GRPO 的 advantage 怎么算

GRPO 不需要 Critic（value function），用**组内相对奖励**作为 advantage：

```python
# 同一个 prompt，采样 G 个输出，计算相对排名
rewards = [r₁, r₂, r₃, ..., r_G]             # G 个输出的奖励
mean_r  = mean(rewards)
std_r   = std(rewards)
advantage_i = (r_i - mean_r) / (std_r + 1e-8)  # 组内 z-score 归一化
```

不需要额外训练一个 Critic 网络，结构简单，这是 GRPO 相比 PPO 的主要优势。

### REINFORCE → PPO → GRPO 演化路径

```
REINFORCE（1992）
    ↓ 问题：单步更新，采样效率低；方差大
    
PPO（2017）
    ↓ 改进：importance ratio 让同一批数据多次更新
    ↓       clip 防止更新步长过大
    ↓ 问题：需要 Critic（value function），参数量翻倍，训练复杂
    
GRPO（2024, DeepSeekMath）
    ↓ 改进：去掉 Critic，用 group-relative reward 代替 advantage
    ↓       保留 importance ratio + clip（多步更新时）
    ↓ 单步退化：π_old = π_new → ratio = 1 → 等价于 REINFORCE + group advantage
```

### 对应到 R3 代码

```python
# R3 的实际计算（单步更新时）
loss = -(log_prob_per_image * cur_image_advantages).sum() / total_image_num

# 其中 cur_image_advantages 来自 GRPO 的 group-relative 归一化：
# advantage_i = (reward_i - mean(rewards_in_group)) / std(rewards_in_group)
```

这就是完整的 **REINFORCE with group-relative baseline**，数学上等价于单步 GRPO。

---

## 七、参考链接

- TRL PR #2565 作者解释：https://github.com/huggingface/trl/pull/2565#issuecomment-2595837761
- TRL Issue #2608 最终讨论：https://github.com/huggingface/trl/issues/2608
- R1-V Issue #174 社区争论：https://github.com/StarsfieldAI/R1-V/issues/174
- flow_grpo 多步实现参考：https://github.com/yifan123/flow_grpo
- DeepSeekMath 论文 Eq.(17)：单步 GRPO 退化为标准 policy gradient
