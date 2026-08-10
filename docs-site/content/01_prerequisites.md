# 01. 必要前置知识

本章不试图完整教授 Transformer、强化学习或分布式系统，而是建立阅读 verl 源码所需的最小词汇和数学直觉。

## 1.1 LLM 在训练框架眼中的样子

### Token 与序列

模型不直接处理字符串，而是处理 token ID：

```text
"17 * 23 是多少？"
    ↓ tokenizer / chat template
[151644, 872, 25, 220, 16, 22, ...]
```

这里的 token ID 只用于说明“文本会被离散化”，并不对应一个承诺的模型输出；具体数字会随 model、tokenizer 和 chat template 改变。

设词表大小为 $V$，一条长度为 $L$ 的序列通常有：

- `input_ids`: $[L]$
- batch 后的 `input_ids`: $[B,L]$
- 模型输出 logits: $[B,L,V]$

chat template 还会插入角色标记、消息边界和工具 schema。对于当前 V1 Agent Loop，dataset 主要提供 `raw_prompt`；真正的 chat-template/tokenization 位于 rollout/Agent Loop 一侧，而不是假定 dataset 已经给出最终 `input_ids`。trainer 也会从 [actor model config 初始化 tokenizer/processor](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L646-L659)，因此不要把上面的 ID 当成跨实验常量。

### Logits、概率与 log-prob

模型在位置 $t$ 输出对下一 token 的 logits：

$$
z_t \in \mathbb{R}^{V}
$$

**公式含义：** 在序列的第 $t$ 个位置，模型会为词表中的每个候选 token 给出一个实数分数；这些分数组成向量 $z_t$，还不是概率。

**符号说明：**

- $z_t$ 是第 $t$ 个位置的 logits 向量；下标 $t$ 表示序列位置或生成时刻。
- $\in$ 表示“属于”。
- $\mathbb{R}$ 表示全体实数，$\mathbb{R}^{V}$ 表示由 $V$ 个实数组成的向量空间；这里的上标 $V$ 表示维数，不是乘方。
- $V$ 是词表大小，所以 $z_t$ 恰好有 $V$ 个分量。

softmax 后得到概率：

$$
\pi_\theta(a_t\mid s_t)
= \frac{\exp z_{t,a_t}}{\sum_{v=1}^{V}\exp z_{t,v}}
$$

**公式含义：** softmax 先把每个 logit 取指数，再用所有候选 token 的指数值之和做归一化，得到模型在状态 $s_t$ 下选择 token $a_t$ 的概率。

**符号说明：**

- $\pi_\theta$ 是参数为 $\theta$ 的策略，也就是当前 actor 模型给出的概率分布；$\theta$ 代表模型的全部可训练参数。
- $a_t$ 是时刻 $t$ 选择的动作，在 token 级语言生成中就是所选的下一个 token；$s_t$ 是做出选择时已经可见的完整上下文。
- $\mid$ 表示“在……条件下”，所以 $\pi_\theta(a_t\mid s_t)$ 是已知 $s_t$ 时选择 $a_t$ 的条件概率。
- $z_{t,a_t}$ 是位置 $t$ 上候选 $a_t$ 对应的 logit；两个下标依次表示位置和候选 token。
- $\exp x$ 表示指数函数，即以自然常数为底的 $x$ 次方；它把任意实数变成正数。
- $\sum_{v=1}^{V}$ 表示让词表索引 $v$ 从 $1$ 遍历到 $V$ 并求和；$z_{t,v}$ 是第 $v$ 个候选 token 的 logit。分母因此汇总了整个词表，保证所有候选概率之和为 $1$。

RL 训练通常使用 log-prob：

$$
\log \pi_\theta(a_t\mid s_t)
$$

**公式含义：** 这就是模型为动作 $a_t$ 给出的概率取对数后的值。动作概率越大，log-prob 越接近 $0$；概率越小，log-prob 越负。

**符号说明：**

- $\log$ 表示自然对数，它把概率的连乘转换为 log-prob 的相加。
- $\pi_\theta(a_t\mid s_t)$ 仍表示参数为 $\theta$ 的策略在状态 $s_t$ 下选择动作 $a_t$ 的概率；下标 $t$ 表示当前生成时刻，$\mid$ 表示条件关系。

原因包括：连乘变连加、数值更稳定，而且 policy-gradient 的核心比率可以写成：

$$
r_t(\theta)
= \frac{\pi_\theta(a_t\mid s_t)}
       {\pi_{\theta_\text{old}}(a_t\mid s_t)}
= \exp\left(
    \log\pi_\theta(a_t\mid s_t)
    -\log\pi_{\theta_\text{old}}(a_t\mid s_t)
  \right)
$$

**公式含义：** $r_t(\theta)$ 比较当前策略与旧策略对同一个动作给出的概率。比率大于 $1$ 表示当前策略提高了该动作的概率，小于 $1$ 表示降低了概率；最右侧说明“概率相除”等价于“log-prob 相减后再取指数”。

**符号说明：**

- $r_t(\theta)$ 是时刻 $t$ 的 probability ratio，并且会随当前参数 $\theta$ 改变；这里的 $r_t$ 是 PPO 比率，不是后文表示即时奖励的同名符号。
- 分子 $\pi_\theta(a_t\mid s_t)$ 来自当前策略，分母 $\pi_{\theta_\text{old}}(a_t\mid s_t)$ 来自固定的旧策略；二者比较的是相同状态 $s_t$ 下的相同动作 $a_t$。
- $\theta_\text{old}$ 表示采样或本轮更新锚点所使用的旧参数；下标 $\text{old}$ 是版本标签，不是乘法。
- $\exp(\cdot)$ 是指数函数，圆括号内的 $\log$ 差值是当前动作 log-prob 减去旧动作 log-prob。
- 两个等号表示三种写法完全等价：比率定义、概率分式和 log-prob 差的指数。

这里至少要区分四个 policy 概念：

- **rollout/behavior policy**：推理 backend 真正采样 trajectory 时的策略，记录为 `rollout_log_probs`；
- **old/proximal policy**：本批 actor update 的固定锚点，记录为 `old_log_probs`；
- **reference policy**：通常是冻结的 SFT 模型，用于 KL 约束；
- **current policy**：正在反向传播的 actor。

理想 on-policy PPO 中 rollout 与 old 是同一个策略。当前 V1 [默认关闭 bypass mode](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/config/algorithm/rollout_correction.yaml#L19-L20)，因此 training actor 会在更新前重算 `old_log_probs`。它与 rollout backend 记录的行为概率在概念和来源上仍需区分；即使名义权重相同，跨 backend 的数值也不保证完全一致。只有 bypass 路径才直接令 `old_log_probs = rollout_log_probs`。两条路径可直接对照 [`_compute_old_log_prob()`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L1479-L1515)。

### Prompt 与 Response

对 decoder-only LLM，可以把序列写成：

```text
[ prompt tokens ][ response tokens ]
```

训练只希望在 actor 采取的 response action 上计算 policy loss。于是常见张量形状是：

```text
input_ids:       [B, L_prompt + L_response]
attention_mask:  [B, L_prompt + L_response]
responses:       [B, L_response]
response_mask:   [B, L_response]
log_probs:       [B, L_response]
advantages:      [B, L_response]
```

### 三种 mask

不要只看到“mask”就当作 padding：

1. `attention_mask`：模型 forward 时，哪些位置是真实上下文；
2. rollout 阶段的 `response_mask`：response 中哪些 token 是模型生成/action；
3. 训练阶段的 effective response/loss mask：实际聚合 loss 时纳入哪些 token，初始通常复制 action mask，也可能再被 rejection 等逻辑修改。

Tool Agent 中尤其重要：工具返回也是真实上下文，所以 attention 可以看见它；但它不是模型 action，因此 policy loss 不应训练它。

## 1.2 把 LLM 生成看成强化学习

### MDP 映射

强化学习常用 Markov Decision Process：

- 状态 $s_t$
- 动作 $a_t$
- 状态转移 $P(s_{t+1}\mid s_t,a_t)$
- 奖励 $r_t$
- 策略 $\pi(a_t\mid s_t)$

映射到 LLM：

| RL | LLM / Agent |
|---|---|
| 状态 $s_t$ | prompt 加上目前为止的 token、工具 observation |
| 动作 $a_t$ | 模型选择的下一个 token，或更高层的 tool call |
| 策略 $\pi_\theta$ | actor LLM |
| 环境 | tokenizer 规则、tool、sandbox、用户模拟器等 |
| trajectory | 一次完整回答或多轮交互 |
| reward | 正确性、格式、偏好、安全性等评分 |

严格来说，语言生成中的状态是完整历史；因为这个历史可见，token-level 展开仍可按 MDP 形式处理。

### Return 与 Advantage

从时刻 $t$ 开始的折扣回报：

$$
G_t = \sum_{k=0}^{T-t-1}\gamma^k r_{t+k}
$$

**公式含义：** 从时刻 $t$ 开始，把本步及之后直到轨迹结束的奖励相加；越远的奖励会被折扣得越小，结果就是时刻 $t$ 的 return。

**符号说明：**

- $G_t$ 是从时刻 $t$ 开始计算的累计折扣回报；下标 $t$ 指回报的起点。
- $\sum_{k=0}^{T-t-1}$ 是求和算子：偏移量 $k$ 从 $0$ 取到 $T-t-1$，因此会覆盖奖励 $r_t$ 到最后一步奖励 $r_{T-1}$。
- $T$ 是轨迹的总时间步数或结束位置，$k$ 表示相对当前时刻向未来走了多少步。
- $\gamma$ 是 discount factor，通常取 $0$ 到 $1$ 之间；$\gamma^k$ 是它的 $k$ 次方，表示第 $k$ 个未来奖励的权重。
- $r_{t+k}$ 是时刻 $t+k$ 获得的即时奖励；下标中的加法表示当前时刻之后第 $k$ 步。

value function 预测从当前状态出发的期望回报：

$$
V^\pi(s_t)=\mathbb{E}[G_t\mid s_t]
$$

**公式含义：** value function 不是预测某一次轨迹必然得到多少分，而是预测处于状态 $s_t$ 后继续按策略 $\pi$ 行动时，平均能够获得多少 return。

**符号说明：**

- $V^\pi(s_t)$ 是状态 $s_t$ 的 value；这里的 $V$ 是 value function，不是前文的词表大小。上标 $\pi$ 表示后续遵循策略 $\pi$，不是乘方。
- $\mathbb{E}$ 是期望算子，可以直观理解为对所有可能后续轨迹的结果取加权平均。
- 方括号中的 $G_t$ 是从时刻 $t$ 开始的累计折扣回报。
- $\mid s_t$ 表示期望是在“当前状态已知为 $s_t$”这个条件下计算；$\mid$ 是条件符号。

advantage 衡量某个实际动作比“当前状态下的平均水平”好多少：

$$
A_t = Q^\pi(s_t,a_t)-V^\pi(s_t)
$$

**公式含义：** advantage 用“选择具体动作 $a_t$ 后的预期回报”减去“只知道当前状态时的平均预期回报”，衡量这个动作比策略的通常表现好或差多少。

**符号说明：**

- $A_t$ 是时刻 $t$ 的 advantage；下标 $t$ 对应这一步的状态和动作。
- $Q^\pi(s_t,a_t)$ 是 action-value function：先在状态 $s_t$ 执行动作 $a_t$，之后按策略 $\pi$ 行动时的期望回报。
- $V^\pi(s_t)$ 是只给定状态 $s_t$、随后按策略 $\pi$ 行动时的期望回报，充当比较基线。
- 两处上标 $\pi$ 都是“使用策略 $\pi$”的标签，不表示乘方；减号得到的是具体动作相对基线的差值。

直觉上：

- $A_t > 0$：提高这个 token/action 的概率；
- $A_t < 0$：降低它的概率；
- $A_t \approx 0$：这个动作没有明显优劣信号。

### Policy Gradient

最基础的策略梯度目标具有如下形态：

$$
\nabla_\theta J(\theta)
\approx
\mathbb{E}_t\left[
A_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right]
$$

**公式含义：** 对许多训练样本取平均：先看动作的 advantage 是正还是负，再按该动作 log-prob 对模型参数的敏感方向调整参数，从而让高 advantage 动作更可能、低 advantage 动作更不可能。

**符号说明：**

- $J(\theta)$ 是希望最大化的策略训练目标，$\theta$ 是当前模型参数。
- $\nabla_\theta$ 是对 $\theta$ 求梯度的算子；$\nabla_\theta J(\theta)$ 是能让目标上升最快的参数方向。
- $\approx$ 表示实际训练通常用有限采样得到的估计值，而不是精确算出所有可能轨迹上的梯度。
- $\mathbb{E}_t$ 表示对采到的时间步或 token 样本 $t$ 取期望，在代码中通常对应求平均。
- $A_t$ 是该动作的 advantage，它作为系数缩放并决定梯度方向。
- $\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ 表示动作 $a_t$ 的 log-prob 对参数 $\theta$ 的梯度；$s_t$ 是当前状态，$\mid$ 表示条件关系。
- 方括号内相邻的两项表示相乘，外层期望再汇总所有样本的贡献。

把它读成一句话：

> advantage 决定“向上还是向下推”，log-prob 的梯度决定“怎样改变参数才能改变这个 action 的概率”。

## 1.3 为什么需要 PPO

如果直接用同一批数据做很大的 policy update，新策略可能离旧策略太远，训练会不稳定。PPO 在 surrogate objective 中裁剪 probability ratio 带来的继续获利空间：

$$
L^{\text{clip}}(\theta)
= \mathbb{E}_t\left[
\min\left(
r_t(\theta)A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

**公式含义：** PPO 对每个样本同时计算未裁剪和已裁剪的 surrogate objective，并取两者中较小的一个再求平均。这样，当概率比率已经变化过大时，继续把它推得更远通常不会继续增加这个近似目标。

**符号说明：**

- $L^{\text{clip}}(\theta)$ 是参数 $\theta$ 下的 PPO clipped surrogate objective；上标 $\text{clip}$ 是目标名称，不是乘方。
- $\mathbb{E}_t$ 表示对训练样本或时间步 $t$ 取平均。
- $\min(x,y)$ 取两个候选值中较小的一个；这里两个候选分别是未裁剪的 $r_t(\theta)A_t$ 和裁剪后的比率乘 $A_t$。
- $r_t(\theta)$ 是当前策略概率除以旧策略概率得到的比率，$A_t$ 是该动作的 advantage；两者相邻表示相乘。
- $\operatorname{clip}(x,l,u)$ 把 $x$ 限制到下界 $l$ 与上界 $u$ 之间：过小就取下界，过大就取上界。
- $\epsilon$ 是一个正的裁剪超参数；$1-\epsilon$ 和 $1+\epsilon$ 分别是允许比率变化的下界与上界。

其中：

```text
r_t = exp(clamp(current_log_prob - old_log_prob, -20, 20))
```

数学定义是 $\exp(\text{current}-\text{old})$；上面展示的是 verl 的数值实现，它先把 log-ratio 裁到 $[-20,20]$ 再取指数，避免指数溢出，也会让超出区间的实现 ratio 饱和。随后 PPO 的 ratio clip 会截住该样本继续改善 surrogate objective 的激励。它不是对参数或最终 ratio 的硬投影，不能保证训练后所有 ratio 都落在 $[1-\epsilon,1+\epsilon]$。实现里通常最小化负目标，因此代码中的 `pg_loss` 符号可能与论文最大化目标相反；当前默认 vanilla loss 对负 advantage 还支持 dual-clip 扩展，后文会单独解释。对应实现见 [`compute_policy_loss_vanilla()`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L1320-L1359)。

PPO 常搭配：

- critic：估计 value；
- GAE：降低 advantage 方差；
- value clipping：稳定 critic；
- entropy bonus：避免策略过早坍缩；
- reference KL：限制 actor 偏离初始模型。

## 1.4 GAE：有 critic 的 advantage

TD residual：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

**公式含义：** TD residual 比较“本步真实奖励加上折扣后的下一状态价值”与“当前状态原先预测的价值”；两者的差就是 critic 在这一步的预测误差。

**符号说明：**

- $\delta_t$ 是时刻 $t$ 的 temporal-difference residual；下标 $t$ 表示当前时间步。
- 这里的 $r_t$ 是时刻 $t$ 收到的即时奖励，不是前面 PPO 的 probability ratio，含义由当前上下文决定。
- $\gamma$ 是 discount factor，用来降低下一状态价值对当前目标的权重。
- $V(s_{t+1})$ 是下一时刻状态 $s_{t+1}$ 的预测价值，$t+1$ 表示紧接着的下一步；$V(s_t)$ 是当前状态的预测价值。
- 加号把即时奖励与折扣后的未来价值合并，减号再扣除当前预测，得到残差。

Generalized Advantage Estimation：

$$
A_t^{\text{GAE}}
= \delta_t
+ (\gamma\lambda)\delta_{t+1}
+ (\gamma\lambda)^2\delta_{t+2}
+ \cdots
$$

**公式含义：** GAE 把当前以及未来多个时间步的 TD residual 加权相加来估计 advantage；距离当前越远，权重会按 $\gamma\lambda$ 的幂次不断衰减。

**符号说明：**

- $A_t^{\text{GAE}}$ 是时刻 $t$ 的 GAE advantage；上标 $\text{GAE}$ 是估计方法的名称，不是乘方。
- $\delta_t$、$\delta_{t+1}$、$\delta_{t+2}$ 分别是当前、下一步和下两步的 TD residual；下标的加数表示向未来移动的步数。
- $\gamma$ 是奖励折扣因子，$\lambda$ 是控制短期与长期 residual 混合程度的 GAE 参数，两者通常都在 $0$ 到 $1$ 之间。
- $(\gamma\lambda)$ 是每向未来一步额外乘上的衰减系数；上标 $2$ 表示连续衰减两次，更一般的上标表示对应的乘方次数。
- $+$ 表示把各步贡献相加，$\cdots$ 表示同样的模式会继续到轨迹结束。

也可从后向前递推：

```python
last_gae = 0
next_value = 0
for t in reversed(range(response_length)):
    delta = reward[t] + gamma * next_value - value[t]
    candidate = delta + gamma * lam * last_gae
    # mask=0 的 observation/padding 不作为 action step：保持递推状态，不重置它
    last_gae = where(response_mask[t], candidate, last_gae)
    next_value = where(response_mask[t], value[t], next_value)
    raw_advantage[t] = last_gae
return_ = raw_advantage + value
advantage = masked_whiten(raw_advantage, response_mask)
```

这段仍是简化伪代码。`return_` 使用 whitening 之前的 `raw_advantage`，而交给 actor 的 `advantage` 会在所有有效 action token 上做 masked whitening；二者不能互换。Tool Agent 的 observation token 具有 `response_mask=0`，但不代表 episode 在那里终止；当前实现跳过 observation 的 value/TD-error，同时保留 `next_value` 和 `last_gae`，从而连接 observation 两侧相邻的模型 action。whitening 只用 mask 内元素计算统计量，下游 loss 也只消费 mask 有效的位置。完整顺序见 [`compute_gae_advantage_return()`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L238-L263)。

## 1.5 GRPO：用组内相对表现代替 learned critic

GRPO 常对同一个 prompt 采样多条 response，然后在组内标准化 reward。简化形式：

$$
A_i = \frac{R_i-\operatorname{mean}(R_{g(i)})}
{\operatorname{std}(R_{g(i)})+\varepsilon}
$$

**公式含义：** 对样本 $i$，先用它的奖励减去同一 prompt 组的平均奖励，再除以组内奖励的标准差。结果表示该样本比同组平均水平好或差了多少个“组内波动单位”。

**符号说明：**

- $A_i$ 是第 $i$ 条 response 样本的组内标准化 advantage；下标 $i$ 是样本编号。
- $R_i$ 是样本 $i$ 的 reward；大写 $R$ 在这里表示整条 response 的序列级奖励。
- $g(i)$ 是“样本 $i$ 属于哪个 prompt 组”的映射，$R_{g(i)}$ 表示该组内所有 response 的 reward 集合。
- $\operatorname{mean}$ 是取算术平均值的算子，所以分子衡量样本奖励与组均值的差。
- $\operatorname{std}$ 是取标准差的算子，用来衡量组内 reward 的波动大小；用它相除会统一不同组的尺度。
- $\varepsilon$ 是一个很小的正数，加在分母中以避免标准差为 $0$ 时除以零；分式横线表示“分子除以分母”。

例如，同一 prompt 的三个 reward 是：

```text
[0, 1, 1]
mean = 2/3
```

则失败样本得到负 advantage，两个成功样本得到正 advantage。随后 sequence-level advantage 通常会广播到该 response 的有效 action token。

GRPO 不需要一个 learned critic 来提供 baseline，但仍要解决：

- 每个 prompt 必须能识别自己的 sample group；
- 组内样本数量和 reward 方差会影响信号；
- 全部 reward 相同时，标准化后的学习信号可能为零；
- PPO-style policy loss 仍需 current/old log-prob；
- 如果启用 KL，还需 reference log-prob 或其他 KL 计算方式。

## 1.6 KL 约束

RLHF 常希望 actor 不要偏离 reference model 太远。KL 的理想形式是：

$$
D_{KL}(\pi_\theta\|\pi_{ref})
= \mathbb{E}_{a\sim\pi_\theta}
\left[\log\pi_\theta(a|s)-\log\pi_{ref}(a|s)\right]
$$

**公式含义：** 这个 KL divergence 衡量当前策略与 reference policy 的概率分布差异：在当前策略可能采取的动作上，平均比较两者给同一动作的 log-prob。值越小，两个分布越接近；它有方向性，交换两侧策略通常会得到不同结果。

**符号说明：**

- $D_{\mathrm{KL}}$ 是 Kullback–Leibler divergence；下标 $\mathrm{KL}$ 是名称缩写。它衡量分布差异，但不是对称的普通距离。
- $\pi_\theta$ 是参数为 $\theta$ 的当前策略，$\pi_{\mathrm{ref}}$ 是 reference policy；下标 $\mathrm{ref}$ 是“参考模型”的标签。
- $\|$ 用来分隔“被比较的当前分布”和“作为参照的分布”，不是绝对值符号；左、右顺序决定 KL 的方向。
- $\mathbb{E}_{a\sim\pi_\theta}$ 表示对从当前策略 $\pi_\theta$ 采样的动作 $a$ 取期望；$\sim$ 表示“服从或采样自这个分布”。
- $a$ 是候选动作或 token，$s$ 是给定状态；$a\mid s$ 表示在状态 $s$ 下选择动作 $a$。
- $\log\pi_\theta(a\mid s)$ 与 $\log\pi_{\mathrm{ref}}(a\mid s)$ 分别是当前策略和参考策略对同一动作给出的 log-prob，减号得到该动作上的 log-prob 差，再由外层期望求平均。

在 sampled token 上会使用不同 estimator。框架里常见两种放置方式：

1. **KL in reward**：从 token reward 中减去 KL penalty，再估 advantage；
2. **KL loss**：policy loss 之外另加 KL 项。

理想公式的期望分布是 $a\sim\pi_\theta$，但一批真实 token 是由 rollout/behavior policy 采出来的。当前 reward-side 路径计算 [`old_log_probs` 与 `ref_log_prob` 的 sampled penalty](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/ray_trainer.py#L94-L109)，loss-side 路径才计算 [`current log_prob` 与 `ref_log_prob`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/utils/losses.py#L120-L142)。只有当相应被比较策略也是采样分布，并使用匹配的 estimator 时，样本均值才可直接解释为上面方向的严格 KL；在 decoupled/off-policy 情形下，更准确的叫法是 sampled KL penalty/estimator。

配置决定 reference worker 是否必要，不能简单认为“所有 PPO/GRPO 都必然启动 ref”。

## 1.7 On-policy、off-policy 与 staleness

理想 on-policy 数据来自当前 actor：

```text
用 θ_k 采样 → 用这些样本更新 θ_k → 得到 θ_{k+1}
```

异步系统中可能出现：

```text
rollout 仍用 θ_k 生成
trainer 已更新到 θ_{k+2}
```

这就是 sample staleness。标准 PPO 在 old policy 就是 behavior policy 时，ratio 能容纳有限的策略差异；但当前 decoupled 路径中的 $\exp(\text{current}-\text{old})$ 主要约束 proximal update，并不会自动校正“rollout policy $\to$ old policy”的偏差。后者要靠模型版本阈值、rollout correction/rejection，或明确采用 old=rollout 的 bypass 语义。verl 的同步、colocate async、separate async 模式，本质上是在吞吐、显存复用和 staleness 之间取舍。

## 1.8 分布式训练最小知识

### Rank 与 Process Group

分布式程序通常启动多个进程，每个进程有一个全局 `rank`。一组进程通过 process group 执行 collective：

- all-reduce：聚合梯度/统计；
- all-gather：收集分片；
- reduce-scatter：规约后再分片；
- broadcast：从一个 rank 发给其他 rank。

### Data Parallel

每个 rank 处理不同数据。朴素 DP 每张 GPU 都持有完整参数；FSDP，以及不同 stage 的 ZeRO，会分片其中一部分或全部参数、梯度和 optimizer state，从而降低单卡显存。

### Tensor Parallel

把同一层的大矩阵沿维度切到多张 GPU。每个 token 的一次 forward 需要 TP rank 协作。vLLM rollout 常用 `tensor_model_parallel_size` 一类配置。

### Pipeline Parallel

把不同层放在不同 stage，micro-batch 像流水线一样通过各 stage。Megatron 类后端常组合 TP、PP、DP。

### Sequence / Context Parallel

沿序列维度切分计算，长上下文时有用。具体支持取决于 engine 与配置。

### 为什么 rollout 与 training 使用不同引擎

训练需要保存激活、反向传播、optimizer state；生成需要高效 KV cache、continuous batching 和低延迟调度。它们优化目标不同：

```text
Training engine: 吞吐 + backward + 分片 optimizer
Rollout engine:  generation throughput + KV cache + request scheduling
```

因此“同一个 actor 模型”在物理上经常有训练态和推理态两份表示，框架必须同步权重。

## 1.9 Ray 最小知识

Ray actor 可以理解成带状态的远程 Python 进程：

```python
import ray

@ray.remote(num_gpus=1)
class Worker:
    def compute(self, x):
        return x * 2

w = Worker.remote()
future = w.compute.remote(21)
result = ray.get(future)
```

在 verl 中：

- launcher/Ray driver 负责启动并等待 controller；controller 决定训练阶段的调用顺序；
- remote worker 持有模型、optimizer 或 rollout server；
- resource pool / placement group 决定它们放到哪些节点和 GPU；
- worker group 把一次逻辑调用分发给多个 rank。

注意：Ray 负责进程和资源调度，不等于 PyTorch distributed。Ray 把进程放好后，worker 内部仍会建立 NCCL/process group 完成模型并行。

## 1.10 Hydra 最小知识

Hydra 用层级 YAML 组合实验配置。下面只演示 override 语法，并不是一条可直接运行的完整实验命令；真实运行还必须选择 rollout backend，并补齐与 static/dynamic batching 对应的 batch 配置：

```bash
python3 -m verl.trainer.main_ppo \
  data.train_files=/path/train.parquet \
  actor_rollout_ref.model.path=/path/model \
  actor_rollout_ref.rollout.name=vllm \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.rollout.n=3 \
  trainer.total_epochs=1
```

理解三点即可：

1. `defaults` 列表决定配置组组合顺序；
2. `${...}` 是 interpolation，值可能来自其他路径；
3. `_target_` 可让 Hydra 按配置实例化 Python class。

不要只凭 YAML 中看到的字面值判断最终配置。应查看合并后的 resolved config 和 `validate_config()` 的约束。

## 1.11 前置知识自测

1. `old_log_probs` 与 `ref_log_prob` 为什么不能互换？
2. 工具 observation 为什么 `attention_mask=1`，但 `response_mask=0`？
3. GRPO 的 group 是按 batch 随意分组，还是按原始 prompt 分组？
4. Ray placement 与 FSDP collective 各自负责什么？
5. 为什么 training engine 更新后，rollout server 不一定自动得到新参数？

这些概念会在后续章节与具体 class、字段和调用链一一对应。
