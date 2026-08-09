# 10. Reward 与 Advantage：模型究竟在为什么信号学习

这一章只回答一件事：一条 rollout 经过工具调用和奖励函数以后，怎样变成 actor 可以学习的 `advantages`。

读完后，你应该能分清下面五个经常被混用的词：

- **score**：外部评审器对整条回答打出的原始分数。
- **reward**：真正交给 RL 算法的逐 token 信号；它可能已经加入 KL 惩罚等 shaping。
- **return**：从某个时刻开始，未来 reward 的折扣和。
- **value**：critic 对未来 return 的预测。
- **advantage**：某个动作比 baseline “好多少”，也是 policy gradient 最直接使用的权重。

本章以当前默认的 V1 trainer 为主。优势估计的公共入口仍然在
[`ray_trainer.py::compute_advantage`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/ray_trainer.py#L187-L282)，具体公式集中在
[`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L215-L1118)。

---

## 1. 先把语言模型看成一个 RL 环境

对自回归语言模型来说：

- 状态 $s_t$：到当前位置为止的完整上下文。
- 动作 $a_t$：模型生成的下一个 token。
- 策略 $\pi_\theta(a_t\mid s_t)$：模型对下一个 token 的概率分布。
- 轨迹 $\tau$：从 prompt 开始，到最终回答结束的整个交互过程。
- 奖励 $R(\tau)$：正确性、格式、工具调用质量等评分。

Tool Agent Loop 中还有一种很重要的内容：**环境 observation**。例如：

```text
prompt
  -> assistant: 调用 search(...)       # 模型动作
  -> tool:      搜索结果                # 环境 observation
  -> assistant: 根据搜索结果给出答案    # 模型动作
```

工具返回也必须进入模型上下文，否则下一轮 assistant 看不到它；但工具返回不是模型采样的动作，因此不应训练模型去“生成”这些 token。

verl 用 `response_mask` 区分二者。`AgentLoopOutput` 明确定义了：模型生成 token 为 1，tool response token 为 0，见
[`agent_loop.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/experimental/agent_loop/agent_loop.py#L90-L108)。

一个简化序列可以写成：

```text
responses:      [调用工具的 token] [工具返回 token] [最终回答 token] [padding]
attention_mask:  1 1 1 1 1 1       1 1 1 1          1 1 1 1        0 0
response_mask:   1 1 1 1 1 1       0 0 0 0          1 1 1 1        0 0
```

对应实现可直接看
[`AgentLoopManager.generate_sequences`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/experimental/agent_loop/agent_loop.py#L581-L600) 和
[`ToolAgentLoop`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/experimental/agent_loop/tool_agent_loop.py#L262-L280)。工具 observation 被追加时，mask 被追加为 0，见
[`tool_agent_loop.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/experimental/agent_loop/tool_agent_loop.py#L433-L449)。

### 1.1 三种 mask 不要混淆

| 字段 | 典型 shape | 1 代表什么 | 主要用途 |
|---|---:|---|---|
| `attention_mask` | `[B, prompt_len + response_len]` | 是真实 token，不是 padding | Transformer attention |
| `response_mask` | `[B, response_len]` | 是 actor 生成的 response token | policy/KL/entropy/advantage 的有效位置 |
| `loss_mask` | 通常与 `response_mask` 相同 | 需要反向传播的 token | engine 内部 loss 计算 |

V1 Agent Loop 当前直接令 `loss_mask = response_mask`，见
[`agent_loop_tq.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/agent_loop_tq.py#L177-L203)。因此：

> 工具 observation 虽然参与前向计算、改变后续状态，但其位置不会产生 policy loss。

---

## 2. 一条训练样本中与 RL 相关的字段

设实际轨迹数为 $B$，padding 后最大 response 长度为 $T$。

| 字段 | shape | 含义 |
|---|---:|---|
| `responses` | `[B, T]` | assistant token、tool observation 和 padding |
| `response_mask` | `[B, T]` | 只选中 assistant 生成 token |
| `rm_scores` | `[B, T]` | reward manager 产生的逐 token 原始分数 |
| `token_level_scores` | `[B, T]` | trainer 对 `rm_scores` 的统一命名 |
| `token_level_rewards` | `[B, T]` | 可选 KL reward shaping 之后的奖励 |
| `values` | `[B, T]` | critic 在每个 response 位置的预测；GAE 才必需 |
| `advantages` | `[B, T]` | actor 更新时每个动作的权重 |
| `returns` | `[B, T]` | critic target，或为了统一接口保存的占位结果 |
| `old_log_probs` | `[B, T]` | PPO 更新开始前的 actor log-probability |
| `ref_log_prob` | `[B, T]` | 冻结 reference policy 的 log-probability |
| `uid` | `[B]`，非 tensor/对象数组 | 同一个原始 prompt 的分组 ID |

V1 trainer 在优势计算前把 `rm_scores` 映射为 `token_level_scores`，再决定是否加入 KL，见
[`trainer_base.py::_compute_advantage`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L1588-L1629)。

完整变换是：

```text
AgentLoopOutput
    responses + response_mask
           |
           v
reward function / reward model
           |
           v
rm_scores == token_level_scores
           |
           |  可选：减去 beta * KL(token)
           v
token_level_rewards
           |
           |  GAE / GRPO / RLOO / REINFORCE++ / ...
           v
advantages + returns
```

---

## 3. 原始 score 怎样落到 token 上

### 3.1 默认是稀疏的 terminal outcome reward

常见数学题 reward function 只返回一个标量，例如正确为 1、错误为 0。verl 的普通 reward manager 会先创建全 0 的 `[B,T]` tensor，然后把这个标量放在最后一个有效 response 位置：

```python
reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
reward_tensor[i, valid_response_length - 1] = reward
```

真实实现见
[`workers/reward_manager/naive.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/reward_manager/naive.py#L90-L175)。如果异步 reward 已经在 Agent Loop 中算完，`AgentLoopOutput.as_dict()` 也会生成 `rm_scores` 并把分数放到末尾，见
[`agent_loop.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/experimental/agent_loop/agent_loop.py#L116-L147)。

例如一条长度为 5 的回答得分 1：

```text
rm_scores = [0, 0, 0, 0, 1]
```

这并不表示前四个 token 没有学习信号。优势估计器会负责把末端结果向前传播或广播。

### 3.2 reward function 可以返回更多诊断维度

普通 reward function 可以返回：

```python
{
    "score": 0.8,             # 真正写入 rm_scores 的标量
    "accuracy_reward": 1.0,   # 额外统计/可供高级 estimator 使用
    "format_reward": 0.2,
}
```

其中 `score` 是总奖励；其他键进入 `reward_extra_info`。GDPO 会按配置从这些独立 reward 维度计算 advantage，见
[`compute_gdpo_outcome_advantage`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L361-L468)。

### 3.3 tool reward 不会自动加到总 reward

`ToolAgentLoop` 会收集每次工具执行返回的 `tool_reward`，但只是追加到 `extra_fields["tool_rewards"]`，见
[`tool_agent_loop.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/experimental/agent_loop/tool_agent_loop.py#L340-L375)。它不会在框架底层自动执行：

```python
final_reward += sum(tool_rewards)
```

是否以及怎样组合工具奖励，必须由你的 reward function/reward manager 明确定义。这样做是必要的：工具返回的 0.2 可能代表局部正确率、费用、延迟或别的量，框架不能猜它应与最终正确性怎样加权。

### 3.4 DAPO 的 overlong shaping 是一个具体例子

DAPO reward manager 可以在 response 靠近最大长度时加入线性负奖励：

$$
r_{\text{final}} = r_{\text{score}} +
\min\left(-\frac{L-L_{\text{expected}}}{L_{\text{buffer}}}\,c,\;0\right).
$$

实现位于
[`workers/reward_manager/dapo.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/reward_manager/dapo.py#L119-L132)。这是 **reward shaping**：它在优势估计之前直接改变了 reward。

---

## 4. KL reward shaping：`score` 与 `reward` 第一次分开

如果启用：

```yaml
algorithm:
  use_kl_in_reward: true
  kl_penalty: kl
  kl_ctrl:
    type: fixed
    kl_coef: 0.001
```

则逐 token reward 变为：

$$
r_t = s_t - \beta\,\widehat{D}_{KL,t},
$$

其中 $s_t$ 是 `token_level_scores`，$\beta$ 是 KL controller 当前系数。代码对应：

```python
kld = kl_penalty(old_log_probs, ref_log_prob, kl_penalty)
kld = kld * response_mask
token_level_rewards = token_level_scores - beta * kld
```

见 [`apply_kl_penalty`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/ray_trainer.py#L78-L117)。注意两点：

1. KL 只在 `response_mask == 1` 的 assistant token 上扣分，工具 observation 不扣。
2. terminal score 通常只在最后一个位置非零，但 KL penalty 可以分布在所有生成 token 上。

例如：

```text
token_level_scores = [0, 0, 1]
kld               = [0.1, 0.2, 0.3]
beta              = 0.01
token_level_rewards
                   = [-0.001, -0.002, 0.997]
```

KL 也可以直接加到 actor loss，而不改 reward。两种路径的差别在下一章说明。

---

## 5. 为什么需要 advantage

目标是最大化期望回报：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)].
$$

利用 log-derivative trick，可以得到 policy gradient 的核心形式：

$$
\nabla_\theta J(\theta)
=\mathbb{E}\left[\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,A_t\right].
$$

直觉非常简单：

- $A_t>0$：提高该 token 在该上下文中的概率。
- $A_t<0$：降低该 token 的概率。
- $A_t\approx0$：这一步几乎不更新。

如果直接令 $A_t=R(\tau)$，算法仍可能无偏，但方差通常很大。减去一个不依赖当前动作的 baseline 不改变期望梯度，却能显著降低方差：

$$
A_t = \text{return}_t - \text{baseline}_t.
$$

GAE、GRPO、RLOO 的核心差别，就是 baseline 从哪里来。

---

## 6. GAE：由 critic 给每个 token 一个 baseline

配置入口：

```yaml
algorithm:
  adv_estimator: gae
  gamma: 1.0
  lam: 1.0
```

GAE 需要 critic 输出 `values`。当前配置在 `critic.enable` 未显式指定时，`adv_estimator=gae` 会自动启用 critic；非 GAE estimator 默认关闭 critic，见
[`trainer/ppo/utils.py::need_critic`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/utils.py#L96-L107)。

### 6.1 两层公式

先计算一步 TD error：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t).
$$

再把未来 TD error 做指数加权：

$$
A_t^{GAE}
=\delta_t+(\gamma\lambda)\delta_{t+1}
+(\gamma\lambda)^2\delta_{t+2}+\cdots.
$$

递推写法是：

$$
A_t=\delta_t+\gamma\lambda A_{t+1}.
$$

最后给 critic 的 target 是：

$$
\text{return}_t=A_t+V(s_t).
$$

### 6.2 一个完整数值例子

假设只有三个生成 token：

```text
reward r = [0.0, 0.0, 1.0]
value  V = [0.2, 0.3, 0.4]
gamma    = 1.0
lambda   = 0.95
```

从后往前：

```text
delta_2 = 1.0 - 0.4                 = 0.6
A_2     = 0.6

delta_1 = 0.0 + 0.4 - 0.3           = 0.1
A_1     = 0.1 + 0.95 * 0.6          = 0.67

delta_0 = 0.0 + 0.3 - 0.2           = 0.1
A_0     = 0.1 + 0.95 * 0.67         = 0.7365
```

因此未归一化结果为：

```text
advantages_raw = [0.7365, 0.6700, 0.6000]
returns        = [0.9365, 0.9700, 1.0000]
```

verl 随后会在所有有效 token 上对 advantage 做 `masked_whiten`，所以真正写入 batch 的 `advantages` 会变成近似零均值、单位方差；`returns` 保留未 whiten 的 critic target。

### 6.3 源码逐行对应

```python
for t in reversed(range(gen_len)):
    delta = rewards[:, t] + gamma * nextvalues - values[:, t]
    lastgaelam_ = delta + gamma * lam * lastgaelam
    nextvalues = values[:, t] * mask[:, t] + (1 - mask[:, t]) * nextvalues
    lastgaelam = lastgaelam_ * mask[:, t] + (1 - mask[:, t]) * lastgaelam

returns = advantages + values
advantages = masked_whiten(advantages, response_mask)
```

真实函数是
[`compute_gae_advantage_return`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L215-L263)。`masked_whiten` 使用 mask 内的样本方差，见
[`torch_functional.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/utils/torch_functional.py#L287-L339)。

### 6.4 Tool Agent Loop 中的关键行为

GAE 遇到 `response_mask == 0` 的 observation token 时，不更新 `nextvalues` 和 `lastgaelam`。因此它会**跳过 observation 位置，把前一轮 assistant 动作与后一轮 assistant 动作连接起来**。

例如：

```text
位置:          0          1          2
内容:       assistant    tool      assistant
response_mask: 1          0          1
```

位置 1 的 value/TD error 不进入递推，位置 0 仍可接收到位置 2 的未来信号。这正是源码中“skip values and TD-error on observation tokens”的含义。

参数直觉：

- `gamma` 越小，越不看重遥远 reward。
- `lambda=0` 更接近一步 TD：方差低，但更依赖 critic 是否准确。
- `lambda=1` 更接近 Monte Carlo return：偏差低，但方差高。
- 当前默认 `gamma=1, lam=1`，适合很多短 episode、稀疏 terminal reward 场景，但不是永远最优。

---

## 7. GRPO：用同一 prompt 的其他答案做相对 baseline

GRPO 不训练 critic。对同一个 prompt 采样 $n$ 个回答，形成一个 group。当前实现用 `uid` 分组，入口在
[`compute_advantage`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/ray_trainer.py#L235-L247)。

### 7.1 当前实现的精确公式

先把逐 token reward 相加成整条轨迹的 outcome score：

$$
R_i=\sum_t r_{i,t}.
$$

对 group $g$ 计算均值与**样本标准差**：

$$
\mu_g=\frac{1}{n}\sum_{i\in g}R_i,
\qquad
\sigma_g=\sqrt{\frac{1}{n-1}\sum_{i\in g}(R_i-\mu_g)^2}.
$$

优势是：

$$
A_i=
\begin{cases}
\dfrac{R_i-\mu_g}{\sigma_g+\epsilon},
& \text{norm\_adv\_by\_std\_in\_grpo=True},\\[6pt]
R_i-\mu_g,
& \text{False}.
\end{cases}
$$

这个标量再广播到该轨迹所有 `response_mask == 1` 的 token：

$$
A_{i,t}=A_i\,m_{i,t}.
$$

实现见
[`compute_grpo_outcome_advantage`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L266-L331)。vectorized 版本数学相同，见
[`compute_grpo_vectorized_outcome_advantage`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L334-L358)。

### 7.2 数值例子

同一题采样三个答案，reward 为：

```text
R = [1.0, 0.0, 0.5]
```

样本均值和样本标准差都是 0.5，所以：

```text
A = [(1.0 - 0.5) / 0.5,
     (0.0 - 0.5) / 0.5,
     (0.5 - 0.5) / 0.5]
  = [1.0, -1.0, 0.0]
```

若第一个回答的 mask 是：

```text
response_mask = [1, 1, 0, 0, 1]
```

则其 token-level advantage 是：

```text
advantages = [1, 1, 0, 0, 1]
```

工具 observation 不产生 loss，但前后两段 assistant token 都得到同一个轨迹级 advantage。

### 7.3 GRPO 是 outcome-only

当前函数首先执行 `token_level_rewards.sum(dim=-1)`，然后把一个标量广播到所有动作 token。因此它不会区分“哪个 token 收到了哪一个 process reward”。即使你提供多个 token reward，最终也只看它们的总和。

此外，GRPO 返回的 `returns` 直接等于 `advantages`。这里的 `returns` 只是为了统一 batch 接口，不是 GAE 意义上的折扣回报；通常也没有 critic 消费它。

### 7.4 两个边界条件

1. `rollout.n` 应大于 1。若一个 group 只有一个样本，当前实现使用 `mean=0, std=1`，所以优势会保留原始 score，而不是变成 0。
2. 若 group 内所有 reward 完全相同，则 $R_i-\mu_g=0$，整个 group 没有 policy-gradient 信号。DAPO 的 group filtering 正是用来重新采样这类无区分度 group；V1 实现在
   [`replay_buffer.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/replay_buffer.py#L248-L298)。

### 7.5 V1 multi-output Agent Loop 的特殊规则

如果一个自定义 Agent Loop 为同一 session 返回多个 `AgentLoopOutput`，V1 对 GRPO 只拿每个 session 的**最后一个 output**参与 group-relative 计算，再把得到的标量 advantage 广播回该 session 的其他 outputs。见
[`compute_advantage_for_multi_trajectories`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/utils.py#L148-L217)。普通 `ToolAgentLoop` 通常每个 session 只有一个 output，因此不会感觉到这层处理。

---

## 8. RLOO：baseline 明确排除自己

RLOO 也使用同一 prompt 的多条 rollout，但第 $i$ 条的 baseline 是**其他**回答的平均 reward：

$$
b_i=\frac{1}{n-1}\sum_{j\ne i}R_j,
\qquad
A_i=R_i-b_i.
$$

代码使用等价写法：

$$
A_i=\frac{n}{n-1}(R_i-\mu_g).
$$

实现见
[`compute_rloo_outcome_advantage`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L587-L636)；vectorized 实现在
[`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L831-L866)。

仍用：

```text
R = [1.0, 0.0, 0.5]
```

得到：

```text
A_0 = 1.0 - mean(0.0, 0.5) =  0.75
A_1 = 0.0 - mean(1.0, 0.5) = -0.75
A_2 = 0.5 - mean(1.0, 0.0) =  0.00
```

与 GRPO 的当前实现相比：

- RLOO 不除以 group 标准差。
- RLOO baseline 不包含当前样本自身。
- RLOO 仍是 outcome-only，并把标量广播到所有有效动作 token。
- 若 group 只有一个样本，当前代码没有 baseline，会保留原始 score；实际训练仍应令 `rollout.n >= 2`。

---

## 9. REINFORCE++：reward-to-go 加全 batch whitening

`reinforce_plus_plus` 不需要 critic，也不需要 `uid` group baseline。它先计算每个 token 的 reward-to-go：

$$
G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots,
$$

再在整个 batch 的有效 token 上 whiten：

$$
A_t=\frac{G_t-\mu_G}{\sigma_G+\epsilon}.
$$

实现见
[`compute_reinforce_plus_plus_outcome_advantage`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L693-L729)。

如果是单轮回答、`gamma=1` 且只有 terminal reward：

```text
reward  = [0, 0, 1]
returns = [1, 1, 1]
```

每个 token 都获得最终结果，再由 batch-level whitening 产生正负相对信号。

### 9.1 Tool Agent Loop 的重要陷阱

当前 REINFORCE++ 递推在每个位置执行：

```python
running_return = reward[:, t] + gamma * running_return
returns[:, t] = running_return
running_return = running_return * response_mask[:, t]
```

源码注释写的是“Reset after EOS”，但 Tool Agent Loop 的工具 observation 同样满足 `response_mask == 0`。因此对：

```text
内容:          assistant  tool-observation  final-assistant
response_mask:     1             0                 1
reward:            0             0                 1
```

`gamma=1` 时当前代码得到的 raw returns 是：

```text
returns = [0, 1, 1]
```

中间 observation 的 return 会被 loss mask 忽略，而工具调用之前的 assistant token 接收不到最终 reward。也就是说，这个 estimator 当前不会像 GAE 那样跨过 observation 的 0 mask 传播 return。对多轮 tool-agent 训练选择它之前，应先确认这种 credit-assignment 语义正是你想要的。

### 9.2 `reinforce_plus_plus_baseline`

baseline 变体先减去同一 `uid` group 的**包含自身**的均值，再把标量广播到 token，最后在全 batch 上 whiten。实现见
[`compute_reinforce_plus_plus_baseline_outcome_advantage`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L533-L584)。

它与 RLOO 的区别是：

- baseline 包含自身；
- 随后还有全 batch token-level whitening；
- 它不做 group std normalization。

### 9.3 名字叫 REINFORCE++，不代表 policy loss 自动变成 REINFORCE

`algorithm.adv_estimator` 只选择 advantage 算法。actor 的 loss 由另一个配置
`actor_rollout_ref.actor.policy_loss.loss_mode` 决定，默认仍是 `vanilla` PPO clipped loss。

因此：

```yaml
algorithm:
  adv_estimator: reinforce_plus_plus
```

只会选择本节的 reward-to-go/whitening，不会自动把 loss 改成 $-\log\pi\,A$。这一点在下一章展开。

---

## 10. 其他 estimator 的定位

注册表位于
[`AdvantageEstimator`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L88-L110)。初学阶段先掌握 GAE、GRPO、RLOO、REINFORCE++，再阅读下表中的扩展。

| estimator | baseline/变换 | 适用信号 | 入口 |
|---|---|---|---|
| `grpo_vectorized` | 与 GRPO 相同，纯 PyTorch 分组向量化 | outcome | [`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L334-L358) |
| `rloo_vectorized` | 与 RLOO 相同，向量化 | outcome | [`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L831-L866) |
| `remax` | sampled return 减 greedy rollout reward | outcome | [`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L732-L765) |
| `grpo_passk` | 每组仅最佳样本得到 `max - second_max` | Pass@k | [`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L471-L530) |
| `opo` | 按 response length 加权的 group reward baseline | outcome/长度校正 | [`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L639-L690) |
| `gpg` | group-centering 后按非零 reward 比例缩放 | outcome | [`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L768-L828) |
| `gdpo` | 每个 reward 维度分别做 GRPO，再加权和并 whiten | 多维 outcome reward | [`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L361-L468) |
| `optimal_token_baseline` | 用 token/path 方差 proxy 学构造更细 baseline | 单轮高级用法 | [`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L869-L987) |
| `tir_optimal_token_baseline` | 面向多轮轨迹的 optimal-token 版本 | 多轮高级用法 | [`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L988-L1118) |

这些函数被注册并不等于每个 trainer/backend 组合都具备相同的端到端辅助流程。例如 ReMax 还需要额外生成 greedy baseline；阅读高级 estimator 时必须继续检查 trainer 的数据准备路径，而不能只看到 enum 就认为已经完整接线。

---

## 11. 四种核心 estimator 对比

| 特性 | GAE | GRPO | RLOO | REINFORCE++ |
|---|---|---|---|---|
| 需要 critic | 是 | 否 | 否 | 否 |
| 需要同 prompt 多采样 | 否 | 是 | 是 | 否 |
| baseline | $V(s_t)$ | group mean | leave-one-out mean | 无显式 baseline；全 batch whitening |
| token credit | 逐 token 递推 | 轨迹标量广播 | 轨迹标量广播 | reward-to-go |
| 当前可跨 tool observation 传播最终 reward | 是，跳过 mask 0 | 是，整轨迹求和后广播 | 是，整轨迹求和后广播 | 否，mask 0 会 reset |
| `returns` 的含义 | critic target | 等于 advantage 的接口字段 | 等于 advantage 的接口字段 | 未 whiten 的 reward-to-go |

一个实用选择顺序：

1. 想先理解经典 PPO，且能承担 critic 内存：从 `gae` 开始。
2. 只有整条轨迹的可验证 outcome reward，且每题能采样多次：从 `grpo` 开始。
3. 想去掉 group std 对 reward scale 的影响：比较 `rloo` 或关闭 GRPO std normalization。
4. 想用 reward-to-go，又不训练 critic：考虑 `reinforce_plus_plus`，但 tool-agent 多轮 mask 行为必须先验证。

---

## 12. 一个可执行的 GRPO 心智模型

假设：

```yaml
data:
  train_batch_size: 2

actor_rollout_ref:
  rollout:
    n: 3

algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: true
```

一个 trainer step 会围绕两个 prompt 形成两个 group：

```text
uid=A -> trajectory A0, A1, A2 -> rewards [1.0, 0.0, 0.5]
uid=B -> trajectory B0, B1, B2 -> rewards [0.0, 0.0, 0.0]
```

优势：

```text
A group -> [ 1.0, -1.0, 0.0]
B group -> [ 0.0,  0.0, 0.0]
```

然后每个标量被广播到各自所有 assistant token，tool observation 和 padding 位置乘 `response_mask` 归零。下一章的 PPO loss 再用这些 token-level advantage 更新 actor。

---

## 13. 调试 reward/advantage 的检查顺序

遇到“reward 明明正常，但模型不学习”时，按以下顺序看：

1. `rm_scores.sum(-1)` 是否等于你期望的轨迹总分。
2. `token_level_rewards.sum(-1)` 是否因 KL 系数过大而改变符号或量级。
3. `response_mask` 是否正确区分 assistant 与 tool observation。
4. group estimator 的 `uid` 是否真的让同一 prompt 的 $n$ 条 rollout 聚在一起。
5. GRPO/RLOO 的 group 是否全部同分；同分就没有相对信号。
6. `advantages[response_mask.bool()]` 的均值、标准差、正负比例是否合理。
7. 多轮工具训练若使用 REINFORCE++，检查 mask 0 是否意外截断最终 reward。
8. 最后再看 PPO ratio、clip fraction 和 loss；advantage 生成之前的问题，不能靠调 clip ratio 修复。

下一章会把 `advantages` 接到 actor/critic 的实际 loss、mini-batch 和 micro-batch 更新路径上。
