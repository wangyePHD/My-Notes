# LeapAlign：任意步骤的 Flow Matching 偏好对齐

> arXiv: [2604.15311](https://arxiv.org/abs/2604.15311)  
> 标题：*LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories*  
> 作者：Zhanhao Liang, Tao Yang, Jie Wu, Chengjian Feng, Liang Zheng  
> 状态：Accepted by CVPR 2026  
> 整理日期：2026-04-24  
> 相关链接：[arXiv 页面](https://arxiv.org/abs/2604.15311) · [项目主页](https://rockeycoss.github.io/leapalign/)

---

## 一句话总结

LeapAlign 要解决的是：**Flow Matching 模型虽然可以把奖励梯度直接反传回生成过程，但完整轨迹太长，显存开销大、梯度还容易爆炸，所以以往方法很难稳定更新“早期生成步骤”**。而早期步骤偏偏最决定画面的整体布局。  

它的核心办法可以记成一句话：

**把长轨迹压缩成“两跳”可反传轨迹，再用“梯度折扣 + 轨迹相似度加权”让这条短轨迹既稳又有用。**

---

## 这篇论文到底在讲什么？

### 通俗版

想象一张图像从纯噪声慢慢生成出来，就像画家先起大构图，再补局部细节。

- 后面的步骤更像“修细节”
- 前面的步骤更像“定布局、定主体关系、定空间结构”

问题在于，过去很多直接用奖励模型微调 Flow Matching 的方法，虽然也能训，但大多只能安全地改后面几步。原因很简单：

- 如果你让奖励梯度穿过完整生成轨迹，链条太长，显存爆炸
- 梯度一路乘过去，数值也容易失控
- 所以很多方法退而求其次，只更新接近最终图像的后期步骤

这会带来一个直观问题：

**模型会更会“修图”，但不一定更会“构图”。**

比如 prompt 说“左边一只红色小鸟，右边两只蓝色杯子”，如果前面步骤没训好，后面再怎么修，也常常只是把颜色或纹理修漂亮，物体数量、相对位置、全局关系还是容易错。

LeapAlign 的想法很聪明：

1. 先照常跑一遍完整生成轨迹，得到真实中间 latent。
2. 然后别在训练时对整条长轨迹反传，而是从里面随机挑两个时间点，构造一条只有两步的“跳跃轨迹”。
3. 奖励还是看最终真实生成的图像，但梯度只沿这条两步轨迹回传。

这样做的好处是：

- 反传路径从“很多步”变成“固定两步”
- 内存和数值稳定性都好了很多
- 又因为随机挑时间点，所以长期来看，任意生成步骤都有机会被更新

所以这篇论文的本质不是“少采样几步生成图像”，而是：

**训练时用短轨迹代替长轨迹来传递奖励梯度，从而实现对任意步骤，尤其是早期步骤的偏好对齐。**

---

## 论文要解决的核心矛盾

### 1. 直接梯度方法为什么有吸引力？

因为 Flow Matching / Rectified Flow 的采样过程是可微的。  
如果奖励模型也是可微的，那么理论上我们可以直接优化：

$$
\max_\theta r(x_0)
$$

其中 $x_0$ 是模型最终生成图像，$r(\cdot)$ 是奖励模型。

这比 GRPO 一类策略梯度方法更“直接”：

- 不需要把生成过程重新解释成 RL 里的随机策略
- 能直接利用 reward 对图像的梯度
- 优化路径更短，也更贴近“我就是要让最终图像得分更高”

### 2. 为什么现有直接梯度方法不够好？

主要有两个问题：

- **长轨迹反传太贵**：完整生成过程有很多采样步，保存中间激活会吃掉大量显存
- **梯度爆炸**：跨很多步反传时，梯度链条会非常不稳定

所以现有方法常见的折中是：

- 只更新末尾某一步或最后几步
- 或者在中间把某些梯度截断

代价就是：

- 早期步骤学不到
- 步骤间的依赖关系信息被丢掉

而论文强调：**早期步骤决定全局布局，这恰恰是文本对齐里最难、也最关键的部分。**

---

## 直觉图景：LeapAlign 在做什么？

把完整采样轨迹记成：

$$
x_1 \rightarrow \cdots \rightarrow x_k \rightarrow \cdots \rightarrow x_j \rightarrow \cdots \rightarrow x_0
$$

其中：

- $x_1$ 是高斯噪声
- $x_0$ 是最终图像对应的 latent
- $k > j$

LeapAlign 不沿着整条链反传，而是构造两跳：

$$
x_k \rightarrow \hat{x}_{j|k} \rightarrow x_j \rightarrow \hat{x}_{0|j} \rightarrow x_0
$$

可以把它理解为：

- 第一跳：从 $x_k$ 直接“跳”到 $j$ 时刻
- 第二跳：从 $x_j$ 直接“跳”到最终 $0$ 时刻

其中：

- $\hat{x}_{j|k}$ 和 $\hat{x}_{0|j}$ 是模型一步预测出来的“未来 latent”
- $x_j, x_0$ 是完整长轨迹上真实得到的 latent

这个设计有两个关键点：

- 前向数值上依然对齐真实长轨迹
- 反向梯度上只需要穿过两步跳跃路径

所以它既保留了真实轨迹的锚点，又把训练反传的链条极大缩短了。

---

## 数学化讲解

## 1. Flow Matching 基础

Flow Matching 学习一个速度场 $v_\theta(x_t, t)$，让噪声逐步流向数据分布。论文采用的是 rectified flow 设定：

$$
x_t = \alpha_t x_0 + \beta_t x_1
$$

其中在 rectified flow 中：

$$
\alpha_t = 1 - t,\qquad \beta_t = t
$$

因此速度可以理解为：

$$
v = \frac{dx_t}{dt} = x_1 - x_0
$$

模型训练的基本目标是拟合这个速度场：

$$
\mathcal{L}_{\mathrm{fm}}
=
\mathbb{E}\left[\left\|v_\theta(x_t, t) - v\right\|_2^2\right]
$$


---

## 2. 一步跳跃预测

对 rectified flow，一个重要性质是：  
从某个时刻 $k$ 的 latent，可以直接一步预测另一个时刻 $j$ 的 latent：

$$
\hat{x}_{j|k} = x_k - (k-j)\, v_\theta(x_k, k)
$$

同理，也可以从 $x_j$ 一步预测最终图像 latent：

$$
\hat{x}_{0|j} = x_j - j\, v_\theta(x_j, j)
$$


这就是 LeapAlign 能做“跳跃轨迹”的数学基础。

### 直觉理解

普通采样是一步一步走。  
LeapAlign 说：既然速度场学出来了，那我可不可以直接跨过中间很多步，一步估计未来某个位置？

答案是可以。于是训练时就可以把长链条压成两跳。

---

## 3. Latent Connector：前向对齐真实轨迹，反向保留可微路径

如果只用 $\hat{x}_{j|k}$ 和 $\hat{x}_{0|j}$，跳得太狠，可能偏离真实长轨迹。  
所以论文引入了一个很关键的小技巧：**latent connector**。

$$
x_j
=
\hat{x}_{j|k}
+
\operatorname{stop\_gradient}(x_j - \hat{x}_{j|k})
$$

$$
x_0
=
\hat{x}_{0|j}
+
\operatorname{stop\_gradient}(x_0 - \hat{x}_{0|j})
$$


这个式子的妙处在于：

- **前向数值**上，它等于真实的 $x_j$ 和 $x_0$
- **反向传播**时，$\operatorname{stop\_gradient}$ 那一项不传梯度
- 所以梯度会把这两个点当作是由 $\hat{x}_{j|k}$ 和 $\hat{x}_{0|j}$ 生成出来的

也就是说：

- 前向看真实轨迹
- 反向走跳跃轨迹

这正是 LeapAlign 的关键工程设计。

---

## 4. 为什么会有“嵌套梯度”问题？

两步跳跃轨迹反传后，$\frac{\partial x_0}{\partial \theta}$ 不只是两个普通梯度项，还会多出一个跨步骤依赖项。论文把它写成：

$$
\frac{\partial x_0}{\partial \theta}
=
- j \frac{\partial v_\theta(x_j)}{\partial \theta}
- (k-j) \frac{\partial v_\theta(x_k)}{\partial \theta}
+ j(k-j)
\frac{\partial v_\theta(x_j)}{\partial x_j}
\frac{\partial v_\theta(x_k)}{\partial \theta}
$$


前两项可以理解成：

- 在 $j$ 这个跳点直接更新一次
- 在 $k$ 这个跳点直接更新一次

最后一项就是所谓的 **nested gradient**，也就是“前一个跳点的更新，会通过后一个跳点继续影响最终输出”。

### 这项为什么重要？

因为它编码了**步骤之间的耦合关系**。  
如果把它完全砍掉，相当于默认：

- 第一步怎么变，和第二步怎么变，彼此独立

这显然不完全对。生成过程里的前后步骤本来就是互相作用的。

### 这项为什么危险？

因为它往往数值更大，更容易导致训练不稳定，甚至梯度爆炸。

---

## 5. Gradient Discounting：不是砍掉，而是打折

过去一些方法为了稳定，会把 nested gradient 直接去掉。LeapAlign 认为这样损失太大，于是提出了**梯度折扣**：

$$
\hat{x}_{0|j}
=
x_j
- j\, v_\theta\!\left(
\alpha x_j + (1-\alpha)\operatorname{stop\_gradient}(x_j)
\right)
$$

这里 $\alpha \in [0,1]$。

这个写法的效果是：

- 前向数值不变，仍然等价于 $v_\theta(x_j)$
- 但反向时，对 $\frac{\partial v_\theta(x_j)}{\partial x_j}$ 的梯度会被乘上一个 $\alpha$

于是上面的 nested gradient 变成：

$$
\alpha\, j(k-j)
\frac{\partial v_\theta(x_j)}{\partial x_j}
\frac{\partial v_\theta(x_k)}{\partial \theta}
$$


### 这一步的意义

- $\alpha = 1$：完全保留 nested gradient，但最不稳定
- $\alpha = 0$：等价于把这条跨步依赖梯度彻底截断
- $0 < \alpha < 1$：保留信息，但降低幅度

论文的消融显示：

- 在 HPSv2.1 奖励下，$\alpha = 0.3$ 最好
- 在 PickScore / HPSv3 下，附录给出的经验值是 $\alpha = 0.1$

所以 LeapAlign 的态度不是“这项梯度有害”，而是：

**这项梯度是有用的，只是太猛，需要打折。**

---

## 6. 奖励目标：为什么用 hinge loss？

如果直接最大化奖励值，模型可能钻奖励模型的空子，出现 reward hacking。  
所以论文采用了一个阈值式目标：

$$
\mathcal{L}_{\mathrm{raw}}
=
\max(0, \lambda - r(x_0))
$$

其中：

- $r(x_0)$ 是奖励模型对最终生成图像的打分
- $\lambda$ 是阈值

它的含义很直白：

- 如果当前奖励还没到阈值，就继续优化
- 如果已经够高了，就不再无脑往上冲

这能缓解训练不稳定和 reward hacking。

论文默认设置里：

- HPSv2.1 的阈值是 $\lambda = 0.55$
- PickScore 的阈值是 $\lambda = 0.4$
- HPSv3 的阈值是 $\lambda = 13.5$

---

## 7. 为什么奖励看真实 $x_0$，而不是跳跃估计 $\hat{x}_{0|j}$？

不少直接梯度方法会把奖励打在一步预测得到的 $\hat{x}_{0|j}$ 上。  
LeapAlign 则坚持把奖励打在真实完整采样得到的 $x_0$ 上。

原因很简单：

- $\hat{x}_{0|j}$ 只是近似，可能更模糊、更有伪影
- 奖励模型在这种近似图像上打分，信号不一定可靠
- $x_0$ 才是真正的最终输出，奖励监督更可信

论文消融也验证了：**用真实 $x_0$ 做 reward input 更好。**

---

## 8. Trajectory-Similarity Weighting：跳得像不像真实轨迹，也要区分对待

并不是所有两跳轨迹都一样靠谱。  
如果某次跳跃预测和真实长轨迹差得太远，那么沿这条轨迹回传的梯度可能会误导训练。

所以论文定义了两个误差：

$$
d_j = \operatorname{mean}\left(|x_j - \hat{x}_{j|k}|\right)
$$

$$
d_0 = \operatorname{mean}\left(|x_0 - \hat{x}_{0|j}|\right)
$$


再定义相似度权重：

$$
w_{\mathrm{sim}}
=
\frac{1}{\max(d_j, \tau) + \max(d_0, \tau)}
$$


最后总损失为：

$$
\mathcal{L}
=
\operatorname{stop\_gradient}(w_{\mathrm{sim}})
\mathcal{L}_{\mathrm{raw}}
$$


### 这部分的直觉

- 轨迹越像真实轨迹，$d_j, d_0$ 越小
- 权重 $w_{\mathrm{sim}}$ 越大
- 这样的样本就更值得信

而 $\tau$ 则是个下限，防止某些几乎重合的样本权重过大。  
论文默认使用 $\tau = 0.1$。

---

## 把整套方法压成一个训练流程

每一轮训练可以概括成下面 6 步：

1. 用当前 Flow Matching 模型正常采样，跑出一条完整轨迹，从 $x_1$ 到 $x_0$。
2. 在整条轨迹里随机抽两个时间点 $k > j$。
3. 用一步跳跃公式从 $x_k$ 预测 $\hat{x}_{j|k}$，再通过 latent connector 对齐到真实 $x_j$。
4. 从 $x_j$ 再跳一步预测 $\hat{x}_{0|j}$，并通过 connector 对齐到真实 $x_0$。
5. 用真实最终图像 $x_0$ 喂给奖励模型，算出 hinge loss，再乘上轨迹相似度权重。
6. 沿这条两步跳跃轨迹反传梯度，并对 nested gradient 用 $\alpha$ 做折扣。

因为 $k, j$ 是随机采样的，所以长期训练后，任意生成步骤都有机会被更新。  
这就是论文题目里 “at any generation step” 的真正含义。

---

## 实验设置速记

### 模型与训练

- 基座模型：FLUX.1-dev
- 默认奖励模型：HPSv2.1
- 优化器：AdamW
- 学习率：$1\times10^{-5}$
- batch size：64
- weight decay：$1\times10^{-4}$
- EMA：0.995
- 训练轮数：300 iterations
- 硬件：16 张 GPU

### 采样设置

- 训练时在线 rollout：720×720，25 steps，CFG = 3.5
- 评测时采样：720×720，50 steps，CFG = 3.5

### 数据

- 一般偏好对齐：HPDv2 的 50K prompts
- 额外也在 MJHQ-30k 上训练
- 组合对齐：50K GenEval prompts

其中组合数据覆盖六类任务，比例为：

$$
7:5:3:1:1:0
$$


分别对应：

- Position
- Counting
- Attribute Binding
- Colors
- Two Objects
- Single Object

---

## 主要实验结果

## 1. 一般偏好对齐结果

在 FLUX 上用 HPSv2.1 做奖励训练时，LeapAlign 相比原始模型和强基线都有稳定提升。

几个关键数字：

- HPSv2.1：**0.4092**，而原始 Flux 是 0.3078，DRTune 是 0.3882
- HPSv3：**15.7678**，而原始 Flux 是 13.5020
- PickScore：**23.7137**，而原始 Flux 是 22.7902，DRTune 是 23.5185
- UnifiedReward-Alignment：**3.4984**
- UnifiedReward-IQ：**3.7244**
- ImageReward：**1.5104**，而原始 Flux 是 1.0455，DRTune 是 1.3562

这说明 LeapAlign 不只是把“训练时同一个奖励”刷高了，而是对多种 out-of-domain evaluator 也有效。

## 2. 组合对齐结果

GenEval 上，LeapAlign 的总分是：

$$
0.7420
$$


对比：

- 原始 Flux：0.6535
- MixGRPO：0.7232
- DRTune：0.7101

更重要的是，它强的地方正是最需要全局布局能力的类别：

- Two Objects：**96.46**
- Colors：**80.59**
- Position：**30.25**
- Attribute Binding：**66.00**

这和论文的核心论点是对得上的：

**能更有效地更新早期步骤，就能更好地改进全局构图和组合关系。**

## 3. 泛化性

论文还额外验证了：

- 换奖励模型也有效：PickScore、HPSv3 都能工作
- 换 prompt 集也有效：HPDv2、MJHQ-30k 都能提升
- 换 Flow Matching 基座也有效：在 Stable Diffusion 3.5 Medium 上也有一致收益

---

## 消融实验说明了什么？

### 1. 梯度折扣确实重要

论文发现：

- $\alpha = 1$ 时，nested gradient 太猛，梯度范数很大，性能反而下降
- $\alpha = 0$ 时，相当于把跨步骤依赖完全扔掉，信息又不够
- **$\alpha = 0.3$** 给出了最好的平衡

这非常像一个经典结论：

**不是所有难训练的梯度都该砍掉，很多时候“缩小它”比“删除它”更对。**

### 2. 两步跳跃是最佳折中

论文比较了 one-step / two-step / three-step leap：

- one-step：已经比一些现有方法强
- two-step：效果和显存最平衡
- three-step：更耗内存，但收益不明显

所以作者最终选了 two-step。

### 3. 一定要覆盖早期步骤

他们把训练时间范围从全范围 $[0,1]$ 缩到 $[0,1/2]$ 后，效果变差。  
这说明：

**早期步骤真的重要，不是可有可无。**

### 4. 随机选 $k,j$ 比固定间距更好

论文还比较了：

- 在 $[0,1]$ 中随机选 $k,j$
- 固定两者间隔为 $1/2$

结果是随机更好。  
这意味着训练看到更多样的跳跃位置组合是有价值的。

---

## 我对这篇论文的理解

我觉得这篇工作的价值，不只是“又提了一个新对齐算法”，而是它抓住了一个非常具体、又很关键的痛点：

**直接梯度后训练的优势大家都知道，但它为什么总是在早期 timestep 这里卡住？**

LeapAlign 给出的回答很务实：

- 不去硬扛整条长轨迹
- 也不把有用的跨步梯度一刀砍掉
- 而是构造一条更短、更稳、仍保留关键信息的反传路径

如果把这篇论文压缩成四个关键词，我会记成：

**two-step leap + latent connector + gradient discounting + similarity weighting**

---

## 这篇论文适合怎么记？

你可以把 LeapAlign 记成下面这个类比：

- ReFL / DRaFT-LV：主要在“后期修细节”
- DRTune：开始尝试改前期，但把一部分跨步依赖梯度切掉了
- LeapAlign：**既训前期，又尽量保留前后步骤之间的作用关系，只是把这部分梯度做了折扣**

所以它的真正贡献不是单点技巧，而是形成了一套完整闭环：

- 让梯度到得了早期步骤
- 让训练足够稳定
- 让短轨迹尽量像真实长轨迹
- 让奖励来自真实最终图像而不是粗糙近似

---

## 局限与边界

论文也明确提到了一些边界：

- 它要求奖励模型可微
- 对 one-step / few-step 生成模型意义没那么大，因为这些模型本来轨迹就很短
- 跳跃轨迹毕竟是近似，所以还得靠轨迹相似度加权来抑制误导性梯度

换句话说，LeapAlign 最适合的舞台是：

**多步 Flow Matching / Rectified Flow 图像生成模型的后训练对齐。**

---

## 最后一句

LeapAlign 的核心洞见可以概括为：

**图像生成的偏好对齐，不能只修最后几步；真正决定“画面像不像 prompt”的，是前面那些定构图的步骤。要想把奖励传到那里，最好的办法不是反传整条长轨迹，而是学会构造一条足够短、足够真、又足够稳的替代路径。**

---

## 参考来源

1. arXiv 摘要页：<https://arxiv.org/abs/2604.15311>
2. arXiv HTML 正文：<https://arxiv.org/html/2604.15311>
3. 项目主页：<https://rockeycoss.github.io/leapalign/>
