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

**这组记号怎样读**：下标 $t$ 表示“第 $t$ 个生成时刻”；$s$、$a$、$\pi$、$\tau$、$R$ 分别是 state（状态）、action（动作）、policy（策略）、trajectory（轨迹）和 reward（奖励）的常用字母。$s_t$ 可以包含 prompt、此前生成的 token 和工具 observation；$a_t$ 只指 actor 采样的 token，不指工具返回。$\theta$ 是模型中所有可训练参数的集合，所以 $\pi_\theta$ 表示“由参数 $\theta$ 决定的策略”。$\pi_\theta(a_t\mid s_t)$ 中的竖线 $\mid$ 读作“在……条件下”，整个式子就是“已经看到状态 $s_t$ 时，模型选择 token $a_t$ 的概率”。$\tau$ 是这些状态、动作和环境 observation 组成的完整序列。$R(\tau)$ 的括号表示 $R$ 把整条轨迹 $\tau$ 映射成一个标量总分；它不同于后文表示单个 token reward 的小写 $r_t$。

Tool Agent Loop 中还有一种很重要的内容：**环境 observation**。例如：

```text
prompt
  -> assistant: 调用 search(...)       # 模型动作
  -> tool:      搜索结果                # 环境 observation
  -> assistant: 根据搜索结果给出答案    # 模型动作
```

工具返回也必须进入模型上下文，否则下一轮 assistant 看不到它；但工具返回不是模型采样的动作，因此不应训练模型去“生成”这些 token。

verl 在 Agent Loop 输出阶段用 `response_mask` 区分二者。`AgentLoopOutput` 明确定义了：模型生成 token 为 1，tool response token 为 0，见
[`agent_loop.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/experimental/agent_loop/agent_loop.py#L90-L108)。

一个简化序列可以写成：

```text
responses:      [调用工具的 token] [工具返回 token] [最终回答 token] [padding]
attention_mask:  1 1 1 1 1 1       1 1 1 1          1 1 1 1        0 0
response_mask:   1 1 1 1 1 1       0 0 0 0          1 1 1 1        0 0
```

对应实现可直接看
[`AgentLoopWorker.generate_sequences`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/experimental/agent_loop/agent_loop.py#L581-L600) 和
[`ToolAgentLoop`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/experimental/agent_loop/tool_agent_loop.py#L262-L280)。工具 observation 被追加时，mask 被追加为 0，见
[`tool_agent_loop.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/experimental/agent_loop/tool_agent_loop.py#L433-L449)。

### 1.1 三种 mask 不要混淆

| 字段 | 典型 shape | 1 代表什么 | 主要用途 |
|---|---:|---|---|
| `attention_mask` | `[B, prompt_len + response_len]` | 是真实 token，不是 padding | Transformer attention |
| `response_mask` | `[B, response_len]` | 初始值 1 表示 actor 生成 token；rollout rejection 后 1 表示仍被保留的 actor token | policy/KL/entropy/advantage 的有效位置 |
| `loss_mask` | `[B, response_len]` | Agent Loop 写入时与原始 `response_mask` 相同 | engine 内部的 token 选择和全局 token 计数 |

V1 Agent Loop 当前直接令 `loss_mask = response_mask`，见
[`agent_loop_tq.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/agent_loop_tq.py#L177-L203)。这是**写入 TransferQueue 时的初始相等关系**，不是整个 step 内永远相等。V1 先用原始 `response_mask` 做 trainer-level KL reward shaping，然后 rollout correction 可以执行 rejection、改写 `response_mask`，最后优势估计器才读取改写后的 mask；相应时序见
[`trainer_base.py::_compute_advantage`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L1598-L1645)，实际覆盖操作见
[`rollout_corr_helper.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/rollout_corr_helper.py#L1049-L1061)。原始 `loss_mask` 不随这一步更新。

因此无 rejection 时两者相同；启用 rejection 后，某个 assistant token 可以满足 `loss_mask == 1` 但 `response_mask == 0`。在当前 PPO 训练路径中，policy、entropy、actor-side KL 和 critic value loss 的**分子有效位置**都显式用 correction 后的 `response_mask` 选出，见
[`workers/utils/losses.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/utils/losses.py#L85-L137) 和
[`workers/utils/losses.py::value_loss`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/utils/losses.py#L179-L203)。`loss_mask` 仍可通过 engine 统计的 `batch_num_tokens` 影响 token-mean 等全局归一化分母；下一章会精确展开这一区别。因此：

> 工具 observation 虽然参与前向计算、改变后续状态，但其位置不会产生 policy loss。rollout rejection 置零的 assistant token 同样不进入 loss 分子，但原始 `loss_mask` 仍可能把它计入 engine 的全局 token 分母。

---

## 2. 一条训练样本中与 RL 相关的字段

设实际轨迹数为 $B$，padding 后最大 response 长度为 $T$。这里 $B$ 是 rollout 展开后 batch 中的轨迹条数，不一定等于原始 prompt 数；$T$ 是每条 response 对齐后的 token 位置数，这条轴上可以同时有 assistant token、tool observation 和 padding。因此表中的 `[B, T]` 表示“共 $B$ 行、每行 $T$ 个位置”。

| 字段 | shape | 含义 |
|---|---:|---|
| `responses` | `[B, T]` | assistant token、tool observation 和 padding |
| `response_mask` | `[B, T]` | Agent Loop 初始只选中 assistant token；rollout correction 后只选中未被 rejection 去掉的 assistant token |
| `loss_mask` | `[B, T]` | 保留 Agent Loop 的原始 assistant-token mask，主要供 engine 选择/计数 |
| `rm_scores` | `[B, T]` | reward manager 输出的逐 token 分数；可能已包含 DAPO overlong 等 manager-side shaping |
| `token_level_scores` | `[B, T]` | trainer 对 `rm_scores` 的统一命名，数值直接复制 |
| `token_level_rewards` | `[B, T]` | trainer-side 可选 KL reward shaping 之后的奖励 |
| `values` | `[B, T]` | critic 在每个 response 位置的预测；GAE 才必需 |
| `advantages` | `[B, T]` | actor 更新时每个 post-correction 保留动作的权重；其他位置由 mask 排除 |
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
V1 RewardLoop
  streaming Agent Loop / colocated post-sampling
           |
           v
rm_scores                     # 可能已有 manager-side shaping
    == token_level_scores
           |
           |  可选：减去 beta * KL(token)
           v
token_level_rewards + original response_mask
           |
           |  可选 rollout correction：只改 mask；已有 rewards 不清零
           v
token_level_rewards + post-correction response_mask
           |
           |  GAE / GRPO / RLOO / REINFORCE++ / ...
           v
advantages + returns
```

---

## 3. Reward manager 输出怎样落到 token 上

### 3.1 默认是稀疏的 terminal outcome reward

常见数学题 reward function 只返回一个标量，例如正确为 1、错误为 0。当前 V1 trainer 总是初始化 `RewardLoopManager`，默认配置使用 8 个 registered `naive` reward worker，见
[`trainer_base.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L326-L336) 和
[`reward.yaml`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/config/reward/reward.yaml#L1-L20)。它有两条执行路径：

1. **Streaming reward**：未启用 reward model 的 rule-based reward，或使用独立 resource pool 的 reward model，会把 reward worker handles 传给 Agent Loop；每个 session 在 rollout 内异步计分。
2. **Colocated reward model**：reward model 开启但没有独立 resource pool 时，handles 为 `None`；trainer 在 replay-buffer sampling 之后成批计分。

分支选择见
[`RewardLoopManager.reward_loop_worker_handles`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/experimental/reward_loop/reward_loop.py#L292-L302)；streaming 分支的计分见
[`AgentLoopWorker._compute_score`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/experimental/agent_loop/agent_loop.py#L961-L1023)，colocated 分支见
[`trainer_base.py::_step_once`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L536-L553) 和
[`trainer_base.py::_compute_reward_colocate`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L1374-L1426)。

两条路径最终都会产生稀疏 terminal `rm_scores`：先创建全 0 的 `[B,T]` tensor，再把标量分数放到最后一个有效 response 位置。等价的核心操作是：

```python
reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
reward_tensor[i, valid_response_length - 1] = reward
```

colocated 分支的当前 V1 实现在
[`RewardManagerBase.assemble_rm_scores`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/experimental/reward_loop/reward_manager/base.py#L61-L82)；streaming 分支先由当前
[`NaiveRewardManager.run_single`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/experimental/reward_loop/reward_manager/naive.py#L34-L99) 算出标量，再由
[`AgentLoopOutput.as_dict`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/experimental/agent_loop/agent_loop.py#L116-L147) 把分数放到未 padding response 的末尾。`verl/workers/reward_manager/*` 是 legacy trainer 路径，不是这一章主讲的 V1 实现。

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

其中 `score` 是写入 `rm_scores` 的总奖励；其他键进入 `reward_extra_info`。算法层的
[`compute_gdpo_outcome_advantage`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L361-L468) 设计为按配置从这些独立 reward 维度计算 advantage。

> **当前 V1 接线限制**：在本章固定的源码快照中，V1 `_compute_advantage` 只从 TransferQueue 取出 `uid`、`response_mask`、`rm_scores`、log-probability 和 `values`，没有把 `prompts`、`attention_mask` 或各 reward component 传给 GDPO，见
> [`trainer_base.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L1588-L1596)。而 GDPO 函数立即需要 `batch["prompts"]`、`batch["attention_mask"]` 和 `non_tensor_batch` 中的 component keys，见
> [`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L412-L433)。因此仅设置 `algorithm.adv_estimator=gdpo` 不能在当前默认 V1 数据路径端到端使用这些维度；需要自定义 V1 接线，或使用会携带这些字段的 legacy 路径。

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

**公式含义**：当惩罚系数 $c\ge0$ 时，最终奖励等于原始任务分数，再加一个“不大于 0”的超长惩罚。回答尚未超过期望长度时，`min` 选到 0；超得越多，负奖励的绝对值越大。$c=0$ 时没有超长惩罚；源码没有校验 $c$ 的符号，因此若误配为负数，上述直觉便不成立。

**符号说明**：$r_{\text{final}}$ 是 shaping 后的最终奖励，$r_{\text{score}}$ 是原始评分；下标 `final` 和 `score` 只是帮助辨认用途。$L$ 是 response 段中 `attention_mask == 1` 的位置数：在 tool-agent 中既含 assistant token，也含 tool observation，但不含 padding，不能把它理解成 `response_mask` 选中的纯 assistant 长度。$L_{\text{buffer}}$ 对应 `overlong_buffer_cfg.len`，必须大于 0；$L_{\text{expected}}$ 等于 `max_resp_len` 减去 $L_{\text{buffer}}$，而源码还要求 `max_resp_len` 不小于 $L_{\text{buffer}}$，但不会进一步验证该配置值是否真的等于 rollout 的 response 容量。$c$ 对应 `penalty_factor`，按惩罚语义应为非负数。$L-L_{\text{expected}}$ 是超出的长度，前面的负号把它变成扣分；分数线表示用超出长度除以缓冲长度，`min` 是取最小值的算子，末尾的 0 保证这一项最多为 0、不会因长度较短而加分。

当前 V1 实现位于
[`experimental/reward_loop/reward_manager/dapo.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/experimental/reward_loop/reward_manager/dapo.py#L32-L119)。这是 **manager-side reward shaping**：它在优势估计之前把 shaped `reward_score` 写入 `rm_scores`，所以后续的 `token_level_scores` 已经包含这项超长惩罚。

---

## 4. Trainer-level KL reward shaping：`token_level_scores` 与 `token_level_rewards` 分开

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
r_t = \mathrm{score}_t - \beta\,\widehat{D}_{KL,t},
$$

**公式含义**：第 $t$ 个 token 真正交给 RL 的奖励，是该位置的 `token_level_scores` 加上 KL shaping 项 $-\beta\widehat{D}_{KL,t}$。对当前 `kl`/`k1` estimator，$\widehat{D}_{KL,t}$ 是带符号的单样本估计：它为正时该 token reward 减小，为负时减去负数、该 token reward 反而增大。因此这不是“每个 token 一律扣分”；惩罚性质只在满足采样条件后的期望意义上成立。

**符号说明**：$t$ 是 token 位置；$r_t$ 是该位置 shaping 后的 reward；$\mathrm{score}_t$ 是该位置的 `token_level_scores`，使用直立英文 `score` 是为了避免与第 1 节表示状态的 $s_t$ 混淆；$\beta$（beta）是 KL controller 的当前系数，按惩罚语义应配为非负数，且越大就越强地限制 old policy 偏离 reference policy。但当前 `KLControlConfig`、fixed controller 和 adaptive controller 都没有校验或 clamp 这个符号，见
[`algorithm.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/config/algorithm.py#L24-L39) 和
[`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L153-L181)；负值配置会破坏上述惩罚直觉。当前配置 `kl_penalty: kl` 精确计算 `old_log_probs - ref_log_prob`，所以 $\widehat{D}_{KL,t}$ 的方向是“old/proximal policy 减 reference policy”，不是反向；当动作确实采样自 old policy 时，其期望对应 forward KL，即 $D_{KL}(\pi_{\mathrm{old}}\|\pi_{\mathrm{ref}})$，双竖线表示从左侧分布到右侧分布的有向比较。$D_{KL}$ 表示 Kullback–Leibler divergence，上方的“帽子”表示它是单个已采样 token 的估计量而非完整分布求和，下标 `KL,t` 分别标出算子种类和 token 位置。单个 token 的估计值可能为负；若数据来自不同的 rollout policy 且没有相应 importance correction，样本均值也不能严格称为这个 old-to-reference KL。

代码对应：

```python
kld = kl_penalty(old_log_probs, ref_log_prob, kl_penalty)
kld = kld * response_mask
token_level_rewards = token_level_scores - beta * kld
```

见 [`apply_kl_penalty`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/ray_trainer.py#L78-L117)。注意两点：

1. KL shaping 发生在 rollout correction 之前，所以这里使用的是 Agent Loop 写入的**原始** `response_mask`：它只选 assistant token，工具 observation 不参与。对默认 `k1` estimator，单 token 的估计值可能为负，所以该位置的 reward 可能减小也可能增大；“惩罚”指满足 on-policy 采样条件后的期望意义。
2. terminal score 通常只在最后一个位置非零，但 KL penalty 可以分布在所有生成 token 上。

随后若 rollout rejection 把某些 assistant 位置的 `response_mask` 改成 0，这些位置已经写入的 KL-shaped `token_level_rewards` 不会被就地清零；优势估计器如何处理它们，取决于各 estimator 是先对 reward 求和，还是按 correction 后的 mask 做递推。第 6、7、9 节会分别说明。

例如：

```text
token_level_scores = [0, 0, 1]
kld               = [0.1, 0.2, 0.3]
beta              = 0.01
token_level_rewards
                   = [-0.001, -0.002, 0.997]
```

这个例子的三个 `kld` 都为正，所以三个位置都扣分。若某位置 `kld=-0.1`、`beta=0.01`，该位置的 shaping 项是 `-0.01 * (-0.1) = +0.001`，reward 会增大 `0.001`。

KL 也可以直接加到 actor loss，而不改 reward。两种路径的差别在下一章说明。

---

## 5. 为什么需要 advantage

目标是最大化期望回报：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)].
$$

**公式含义**：训练要调整模型参数，使模型按当前策略生成许多轨迹时，平均得到的整轨迹奖励尽可能大。

**符号说明**：$J$ 是训练目标函数，$J(\theta)$ 表示它取决于模型参数 $\theta$；$\mathbb{E}$ 是对方括号内的随机结果取期望，也就是“重复采样很多次后的平均值”；$\tau\sim\pi_\theta$ 表示轨迹 $\tau$ 是从策略 $\pi_\theta$ 采样出来的，其中 $\sim$ 读作“服从/采样自”；$R(\tau)$ 是轨迹 $\tau$ 的总奖励。

利用 log-derivative trick，可以得到 policy gradient 的核心形式：

$$
\nabla_\theta J(\theta)
=\mathbb{E}\left[\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,A_t\right].
$$

**公式含义**：把一条回答中每个 token 的“概率应往哪个方向改”累加起来，再对很多轨迹取平均，就得到更新模型参数的方向；优势 $A_t$ 决定第 $t$ 个 token 的更新方向和力度。

**符号说明**：$\nabla_\theta$ 是“对参数 $\theta$ 求梯度”，即寻找让目标变化最快的参数方向；等号左侧 $\nabla_\theta J(\theta)$ 是整个目标的梯度。$\mathbb{E}$ 表示对采样轨迹取期望；$\sum_t$ 表示把所有生成时刻 $t$ 的项相加；$\log$ 是自然对数，它把 token 概率变成 log-probability；$\pi_\theta(a_t\mid s_t)$ 是在状态 $s_t$ 下选择动作 $a_t$ 的概率；$\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ 是这个 log-probability 对参数的梯度；$A_t$ 是第 $t$ 个动作的 advantage，并与前面的梯度相乘。

**等式成立条件**：这里写的是有限时域、on-policy、未折扣（或已把折扣权重吸收到 advantage 定义中）的理想 policy-gradient identity；轨迹需由 $\pi_\theta$ 采样，$A_t$ 需是真实 advantage 或条件无偏估计，环境转移与 reward 也不能有未计入的显式参数依赖。若用 return 减 baseline，baseline 在给定状态后不能依赖当前采样动作，否则减去它会改变期望梯度。实际 PPO 使用 old policy 数据、importance ratio、clipping 和近似/归一化 advantage，优化的是相应的 surrogate，并不是这个等式在每个 batch 上的精确实现；求和也只覆盖当前 `response_mask == 1` 的 actor 动作位置。无 rollout rejection 时这就是全部 assistant 动作；启用 rejection 后只剩 correction 保留的 assistant 动作。

直觉非常简单：

- $A_t>0$：第 $t$ 个动作的优势为正，提高该 token 在当前上下文中的概率；$>$ 表示“大于”。
- $A_t<0$：第 $t$ 个动作的优势为负，降低该 token 的概率；$<$ 表示“小于”。
- $A_t\approx0$：第 $t$ 个动作的优势近似为零，这一步几乎不更新；$\approx$ 表示“近似等于”。

如果直接令 $A_t=R(\tau)$，也就是让每个时刻 $t$ 的优势 $A_t$ 都等于整条轨迹 $\tau$ 的奖励 $R(\tau)$，算法仍可能无偏，但方差通常很大。减去一个不依赖当前动作的 baseline 不改变期望梯度，却能显著降低方差：

$$
A_t = \text{return}_t - \text{baseline}_t.
$$

**公式含义**：一个动作“比通常水平好多少”，等于它实际对应的未来回报减去一个用于比较的通常水平。

**符号说明**：$A_t$ 是时刻 $t$ 的优势；$\text{return}_t$ 是从时刻 $t$ 开始的未来累计 reward；$\text{baseline}_t$ 是同一时刻用作参照的基线。下标 $t$ 表明三个量都针对同一个生成位置；减号表示只保留“超过或低于基线的差值”。LaTeX 命令 `\text{}` 只是让 `return` 与 `baseline` 这些英文标签以普通字体显示。

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

**公式含义**：一步 TD error 衡量“刚拿到的 reward 加上下一个状态的预测价值”与“critic 原先对当前状态的预测”相差多少。$\delta_t>0$ 表示 critic 相对这个一步目标低估，$\delta_t<0$ 表示高估；$|\delta_t|$ 才是预测偏差的大小，竖线表示取绝对值。

**符号说明**：$\delta_t$（delta）是时刻 $t$ 的一步 temporal-difference error；$r_t$ 是该时刻 reward；$\gamma$（gamma）是 0 到 1 之间的折扣因子，用来降低较远未来价值的权重；$V$ 是 critic 的 value 函数，$V(s_t)$ 是当前状态 $s_t$ 的预测价值，$V(s_{t+1})$ 是下一状态的预测价值；下标 $t+1$ 表示下一个生成时刻，相邻的 $\gamma V(s_{t+1})$ 表示两者相乘。加号把即时 reward 与折扣后的下一状态价值合并，最后减去当前预测；轨迹终止后没有下一状态，约定它的 next value 为 0。

再把未来 TD error 做指数加权：

$$
A_t^{GAE}
=\delta_t+(\gamma\lambda)\delta_{t+1}
+(\gamma\lambda)^2\delta_{t+2}+\cdots.
$$

**公式含义**：时刻 $t$ 的 GAE advantage 不只看当前 TD error，还把以后各步的 TD error 加进来。只有当 $|\gamma\lambda|<1$ 时，距离越远，权重才按相同倍率几何缩小；当前默认 `gamma=1, lam=1`，乘积为 1，默认并不衰减。连续同号的 TD error 会累积，异号则可能互相抵消。

**符号说明**：$A_t^{GAE}$ 是时刻 $t$ 的 generalized advantage estimate，右上角 `GAE` 是算法名称而不是乘方；$\delta_t$、$\delta_{t+1}$、$\delta_{t+2}$ 分别是当前、下一步、下两步的 TD error。$\lambda$（lambda）是控制“看多远”的 GAE 参数；$\gamma\lambda$ 是每往后一步共同乘上的权重倍率，绝对值小于 1 才是衰减倍率；右上角 2 表示该倍率连乘两次；$\cdots$ 表示同样规律继续到轨迹结束。

递推写法是：

$$
A_t=\delta_t+\gamma\lambda A_{t+1}.
$$

**公式含义**：从后往前算时，当前 advantage 等于当前 TD error，再加上折扣后的下一步 advantage；它与上面的展开式是同一件事。

**符号说明**：$A_t$ 与 $A_{t+1}$ 分别是当前和下一时刻的 advantage；$\delta_t$ 是当前 TD error；$\gamma\lambda$ 是折扣因子 $\gamma$ 与 GAE 参数 $\lambda$ 的乘积。这里省略上标 `GAE` 只是简写，并没有换成别的 advantage；从轨迹末端开始反推时，末端之后的 advantage 初值设为 0。

最后给 critic 的 target 是：

$$
\text{return}_t=A_t+V(s_t).
$$

**公式含义**：把“比 critic 预测好多少”加回 critic 原预测，就得到训练 critic 时使用的回报目标。

**符号说明**：$\text{return}_t$ 是时刻 $t$ 的 GAE/λ-return critic target，不一定等于简单的 Monte Carlo reward 总和；$A_t$ 是该时刻尚未 whiten 的 advantage；$V(s_t)$ 是 critic 对状态 $s_t$ 的价值预测。`return` 用普通字体显示是标签，下标 $t$ 仍表示 token 时刻。

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

verl 随后会在所有有效 token 上对 advantage 做 `masked_whiten`，所以真正写入 batch 的 `advantages` 会变成近似零均值、单位方差；`returns` 保留未 whiten 的 critic target。这里“白化”是先只用 mask 选中的 advantage 计算带 Bessel 校正的样本方差，再让每个有效值减去均值并除以“样本方差加 `1e-8` 后的平方根”；correction 后 mask 为 0 的 observation、padding 和 rejected assistant token 都不参与统计。第 9 节给出同一 helper 的精确公式和“全 batch 只有一个有效 token”时的异常边界。

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

### 6.4 mask 中断位置的关键行为

GAE 使用 rollout correction 之后的 `response_mask`。遇到 `response_mask == 0` 的位置时，它不更新 `nextvalues` 和 `lastgaelam`。因此它既会**跳过 observation 位置，把前一轮 assistant 动作与后一轮 assistant 动作连接起来**，也会以同样方式跳过被 rejection 置零的 assistant token。

例如：

```text
位置:          0          1          2
内容:       assistant    tool      assistant
response_mask: 1          0          1
```

位置 1 的 value/TD error 不进入递推，位置 0 仍可接收到位置 2 的未来信号。这正是源码中“skip values and TD-error on observation tokens”的行为；在当前 V1 时序下，rejected assistant token 也落入同一 `mask == 0` 分支，其自身 reward/value/TD error 不进入递推，但递推状态会跨过它继续向前传。

参数直觉：

- `gamma` 越小，越不看重遥远 reward。
- `lambda=0` 更接近一步 TD：方差低，但更依赖 critic 是否准确。
- `lambda=1` 更接近 Monte Carlo return：偏差低，但方差高。
- 当前默认 `gamma=1, lam=1`，适合很多短 episode、稀疏 terminal reward 场景，但不是永远最优。

---

## 7. GRPO：用同一 prompt 的组内均值做相对 baseline

GRPO 的 advantage 不使用 critic baseline，因此非 GAE estimator 在 `critic.enable` 未指定时默认关闭 critic；但若显式设置 `critic.enable=true`，trainer 仍会计算 value 并更新 critic，GRPO advantage 本身依然忽略这些 value，不能笼统理解为“GRPO 永远不训练 critic”。对同一个 prompt 采样 $n$ 个回答，形成一个 group；在普通的“每个 session 一个 output”情况下，这里 $n$ 通常就是配置中的 `rollout.n`。当前实现用 `uid` 分组，入口在
[`compute_advantage`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/ray_trainer.py#L235-L247)。

### 7.1 当前实现的精确公式

先把逐 token reward 相加成整条轨迹的 outcome score：

$$
R_i=\sum_t r_{i,t}.
$$

**公式含义**：第 $i$ 条回答的总分，是把它在所有 token 位置上的 shaped reward 不加折扣地全部相加；求和后只保留总数，不再保留 reward 出现在哪个 token 的时间信息。

**符号说明**：$i$ 是 group 内回答的编号；$R_i$ 是第 $i$ 条完整轨迹的 outcome score；$r_{i,t}$ 是第 $i$ 条轨迹在 token 时刻 $t$ 的 reward，逗号把“轨迹编号”和“时间编号”两个下标隔开；$\sum_t$ 是求和算子，表示遍历该轨迹的所有 $t$ 并相加。

对 group $g$ 计算均值与**样本标准差**：

$$
\mu_g=\frac{1}{n}\sum_{i\in g}R_i,
\qquad
\sigma_g=\sqrt{\frac{1}{n-1}\sum_{i\in g}(R_i-\mu_g)^2}.
$$

**公式含义**：第一式算 group 的平均分，把同题回答的平均 reward 当作相对 baseline；这个均值包含当前第 $i$ 条回答自己。第二式算样本标准差，也就是各回答分数围绕平均分的典型波动幅度。

**符号说明**：$g$ 表示由相同 `uid` 标识的 prompt group；$\mu_g$（mu）是 group $g$ 的均值，$\sigma_g$（sigma）是它的样本标准差；$n$ 是组内回答数；$i\in g$ 表示“编号 $i$ 属于 group $g$”，因此 $\sum_{i\in g}$ 是把组内每个回答的 $R_i$ 相加。$\frac{1}{n}$ 表示总和除以样本数。$R_i-\mu_g$ 是第 $i$ 个分数与均值的偏差，右上角 2 表示把偏差平方，使正负偏差都贡献正的离散程度；$\frac{1}{n-1}$ 中使用 $n-1$ 而不是 $n$，表示这里计算的是带 Bessel 校正的样本方差，因此常规公式要求 $n>1$，单样本 group 使用后文说明的特殊处理；根号是平方根算子，把方差变回与 reward 相同的尺度。LaTeX 命令 `\qquad` 只是在两个等式之间留出排版空隙，没有数学运算含义。

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

**公式含义**：开启标准差归一化时，先看第 $i$ 条回答比组平均高或低多少，再除以组内波动尺度；关闭时，只保留它与组平均的原始差值。两行只会按配置选择其中一行。

**符号说明**：$A_i$ 是第 $i$ 条轨迹的标量 advantage；$R_i$ 是其总分；$\mu_g$ 与 $\sigma_g$ 分别是所在 group 的均值和样本标准差；$\epsilon$（epsilon）是很小的正数，用来避免标准差为 0 时除以 0；分数线表示用分子 $R_i-\mu_g$ 除以分母 $\sigma_g+\epsilon$。`cases` 的大括号表示分情况定义；`norm_adv_by_std_in_grpo=True` 与 `False` 是选择两种情况的布尔配置，`[6pt]` 只增加两行之间的排版距离。

这个标量再广播到该轨迹 correction 后所有 `response_mask == 1` 的 token：

$$
A_{i,t}=A_i\,m_{i,t}.
$$

**公式含义**：把轨迹级 advantage 复制到每个 token，再乘 correction 后的 mask；未被 rejection 去掉的 assistant token 保留该值，observation、padding 和 rejected assistant token 变成 0。

**符号说明**：$A_{i,t}$ 是第 $i$ 条轨迹在时刻 $t$ 的 token-level advantage；$A_i$ 是整条轨迹共享的标量 advantage；$m_{i,t}$ 是优势估计时对应位置的 post-correction `response_mask`，保留的 assistant token 为 1，observation、padding 或被 rejection 去掉的位置为 0；相邻的 $A_i\,m_{i,t}$ 表示两者相乘。

实现见
[`compute_grpo_outcome_advantage`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L266-L331)。vectorized 版本数学相同，见
[`compute_grpo_vectorized_outcome_advantage`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L334-L358)。

这里有一个 V1 配置接线差异：公共 dispatcher 只在精确的 `grpo` 分支传入 `norm_adv_by_std_in_grpo`，见
[`ray_trainer.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/ray_trainer.py#L235-L279)。`grpo_vectorized` 落入通用分支，当前没有收到该参数，因此在固定快照中总是使用函数默认值 `True`。也就是说，两者在默认开启标准差归一化时数学相同，但把 `algorithm.norm_adv_by_std_in_grpo` 设为 `false` 只会改变精确的 `grpo`，不会改变 V1 的 `grpo_vectorized`。

### 7.2 数值例子

同一题采样三个答案，reward 为：

```text
R = [1.0, 0.0, 0.5]
```

样本均值和样本标准差都是 0.5。保留默认稳定项 `epsilon=1e-6` 时：

```text
A = [(1.0 - 0.5) / (0.5 + 1e-6),
     (0.0 - 0.5) / (0.5 + 1e-6),
     (0.5 - 0.5) / (0.5 + 1e-6)]
  ~= [0.999998, -0.999998, 0.0]
```

若第一个回答的 mask 是：

```text
response_mask = [1, 1, 0, 0, 1]
```

则其 token-level advantage 是：

```text
advantages ~= [0.999998, 0.999998, 0, 0, 0.999998]
```

工具 observation 不产生 loss；若没有额外 rejection，前后两段 assistant token 都得到同一个轨迹级 advantage。若某个 assistant 位置已被 rejection 置零，则该位置的 advantage 也为 0。

### 7.3 GRPO 是 outcome-only

当前函数首先执行 `token_level_rewards.sum(dim=-1)`，然后才把一个标量乘 correction 后的 `response_mask`。因此它不会区分“哪个 token 收到了哪一个 process reward”。即使你提供多个 token reward，最终也只看它们的总和。

这个求和本身**没有先乘 post-correction `response_mask`**。因此 rejection 置零的 assistant 位置若已经有 reward（包括先前写入的 KL shaping 项），该 reward 仍会进入 $R_i$ 及 group 均值/标准差；rejection 只让该位置不接收广播后的 advantage。sequence-level rejection 把整行 mask 置零时，这一行的 reward 仍可能参与其他行的 group baseline，只是自身所有 token advantage 都归零。RLOO、`reinforce_plus_plus_baseline`、OPO、GPG 等先执行 `token_level_rewards.sum(dim=-1)` 的 outcome estimator 也有同一顺序。

此外，GRPO 返回的 `returns` 直接等于 `advantages`。这里的 `returns` 只是为了统一 batch 接口，不是 GAE 意义上的折扣回报；通常也没有 critic 消费它。

### 7.4 两个边界条件

1. `rollout.n` 应大于 1。若一个 group 只有一个样本，当前实现使用 `mean=0, std=1`，所以开启标准差归一化时优势是 `score / (1 + epsilon)`：近似保留原始 score，而不是变成 0。
2. 若 group 内所有 reward 完全相同，则 $R_i-\mu_g=0$：第 $i$ 条轨迹的总分 $R_i$ 恰好等于 group 均值 $\mu_g$，两者差值为 0，因此整个 group 没有 policy-gradient 信号。V1 内置 `ReplayBuffer`/`ReplayBufferAsync` 的 DAPO group filtering 只有在训练 partition、`algorithm.filter_groups.enable=true` 且配置了非空 `metric` 时才启用；自定义 sampler 自行负责过滤语义。它比较的是每条轨迹 `reward_extra_info[metric]` 中的**指定指标**，不一定是这里的总 reward；任何轨迹缺少该指标都会报错。该指标还必须在 sampling 时可得，即使用 rule-based reward，或令 reward model 使用 resource pool；仅在采样后才计算的 colocated reward model 不满足条件。一个已完成 group 含多条轨迹且该指标的标准差精确等于 0 时，才会被过滤并补采样，这里没有近似容差。实现见
   [`replay_buffer.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/replay_buffer.py#L248-L298)。

### 7.5 V1 multi-output Agent Loop 的特殊规则

如果一个自定义 Agent Loop 为同一 session 返回多个 `AgentLoopOutput`，V1 对 GRPO 只拿每个 session 的**最后一个 output**参与 group-relative 计算，再把得到的标量 advantage 广播回该 session 的其他 outputs。见
[`compute_advantage_for_multi_trajectories`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/utils.py#L148-L217)。普通 `ToolAgentLoop` 通常每个 session 只有一个 output，因此不会感觉到这层处理。

这个 session-final 折叠**只对精确的 `grpo` estimator 生效**。wrapper 对其他 estimator 直接把整批 output rows 交给通用 `compute_advantage`，见上述函数的
[`L167-L176`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/utils.py#L167-L176)。V1 会把每个 output 都以 `{uid}_{session_id}_{index}` 存成一行；当 streaming reward 已在 Agent Loop 内产生 `final_output.reward_score` 时，还会把该最终 reward 复制给同 session 前面的 outputs，见
[`agent_loop_tq.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/agent_loop_tq.py#L172-L203)。因此对 RLOO、`reinforce_plus_plus_baseline`、`rloo_vectorized` 等 group estimator，自定义 multi-output loop 会让同一 session 的多行都参与 `uid` 分组；group size 可能大于 `rollout.n`，而 streaming 分支中同一 session reward 会重复出现。不能把 GRPO 本节的 session-final 语义外推到这些 estimator。

---

## 8. RLOO：baseline 明确排除自己

RLOO 也使用同一 prompt 的多条 rollout，但第 $i$ 条的 baseline 是**其他**回答的平均 reward；这里 $i$ 是当前回答在 group 内的编号：

$$
b_i=\frac{1}{n-1}\sum_{j\ne i}R_j,
\qquad
A_i=R_i-b_i.
$$

**公式含义**：先把除当前回答之外的所有回答分数取平均，得到当前回答的比较基线；再用当前分数减去该基线，得到 leave-one-out advantage。

**符号说明**：$b_i$ 是给第 $i$ 条回答使用的 leave-one-out baseline；$n$ 是组内回答总数，所以排除自己后还剩 $n-1$ 条，常规公式要求 $n\ge2$；$j$ 是遍历其他回答时使用的编号；$j\ne i$ 表示 $j$ 不等于当前编号 $i$；$\sum_{j\ne i}R_j$ 把所有其他回答的总分 $R_j$ 相加，再乘 $\frac{1}{n-1}$ 得到平均值。$A_i$ 是当前回答的 advantage，$R_i$ 是当前总分，减去 $b_i$ 后，正值表示胜过其他回答的平均水平，负值表示低于该水平。`\qquad` 只是排版空格。

代码使用等价写法：

$$
A_i=\frac{n}{n-1}(R_i-\mu_g).
$$

**公式含义**：也可以先算当前分数与“包含自己在内的 group 均值”之差，再乘一个校正倍数；结果与上面的 leave-one-out 写法完全相同。

**符号说明**：$A_i$、$R_i$、$n$、$i$ 与上式含义相同；$\mu_g$ 是包含 group $g$ 全部 $n$ 条回答的均值；$R_i-\mu_g$ 是当前分数相对全组均值的差；$\frac{n}{n-1}$ 是把这个差转换为“相对其他 $n-1$ 条回答均值”的校正因子，括号表示先计算其中的差值再相乘。这个等价式只在 $n>1$ 时成立，也没有像 GRPO 那样再除以 group 标准差。

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
- RLOO 仍是 outcome-only，并把标量广播到 correction 后保留的动作 token；rejection 与 reward 求和的先后顺序见 7.3 节。普通单 output session 中，group 通常有 `rollout.n` 行；自定义 multi-output session 则受 7.5 节的非 GRPO 限制影响，每个 output row 都会参与 baseline。
- 若 group 只有一个样本，普通 `rloo` 实现没有可用 baseline，会在广播前保留原始 score；`rloo_vectorized` 则把 singleton group 的 advantage 明确归零。两者在这个边界上并不等价，实际训练应保证 `rollout.n >= 2`，不要依赖任一静默 fallback。

---

## 9. REINFORCE++：reward-to-go 加全 batch whitening

`reinforce_plus_plus` 不需要 critic，也不需要 `uid` group baseline。它先计算每个 token 的 reward-to-go：

$$
G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots,
$$

**公式含义**：时刻 $t$ 的 reward-to-go，是从当前位置起把现在和未来的 reward 全部累加；只有当 $|\gamma|<1$ 时，越远的 reward 权重才会几何缩小，`gamma=1` 时所有未来 reward 等权。有限轨迹在 episode 结束处停止求和。这个公式描述没有内部 mask 断点时的完整传播；当前实现遇到 post-correction mask 0 会截断，具体见 9.1 节。

**符号说明**：$G_t$ 是从时刻 $t$ 开始的累计未来回报；$r_t$、$r_{t+1}$、$r_{t+2}$ 是当前、下一步和下两步的 reward；$\gamma$ 是折扣因子，$\gamma^2$ 表示折扣因子连乘两次，因此当 $0\le\gamma<1$ 时两步后的 reward 权重更小；$\cdots$ 表示继续按同样规律加到轨迹结束。各项之间的加号表示把不同时间的 reward 汇总。

再在整个 batch 的有效 token 上 whiten：

$$
A_t=\frac{G_t-\mu_G}{\sqrt{v_G+10^{-8}}}.
$$

**公式含义**：将所有有效 token 的 reward-to-go 做全 batch 白化：每个值先减去全体均值，再除以“样本方差加固定稳定项后开平方”的结果，使输出近似以 0 为中心，并把不同 batch 的尺度拉到相近范围。这是源码 `masked_whiten` 的形式；稳定项加在方差里面、开平方之前，并不是在标准差之后再加一个 epsilon。

**符号说明**：$A_t$ 是时刻 $t$ 白化后的 advantage；$G_t$ 是该时刻的 reward-to-go；$\mu_G$ 是当前 batch 所有 post-correction `response_mask == 1` 位置的 $G$ 均值；$v_G$ 是这些有效值带 Bessel 校正的样本方差，直观上就是把每个值与均值之差的平方相加，再除以“有效 token 数减 1”；根号把方差换回与 $G$ 相同的尺度。$10^{-8}$ 是写死的数值稳定常数，分数线表示 $G_t-\mu_G$ 除以整个根号结果；下标 $G$ 表示这些统计量来自 reward-to-go 集合。这里的样本数是全 batch 被 correction 保留的 token 总数，不是 `uid` group 数；若该总数为 1，`masked_var` 会直接抛出 `ValueError`，稳定常数不会绕过这个检查。

实现见
[`compute_reinforce_plus_plus_outcome_advantage`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L693-L729)。

如果是单轮回答、`gamma=1` 且只有 terminal reward：

```text
reward  = [0, 0, 1]
returns = [1, 1, 1]
```

每个 token 都获得最终结果，再由 batch-level whitening 产生正负相对信号。

### 9.1 post-correction mask 0 会截断 credit

当前 REINFORCE++ 递推在每个位置执行：

```python
running_return = reward[:, t] + gamma * running_return
returns[:, t] = running_return
running_return = running_return * response_mask[:, t]
```

源码注释写的是“Reset after EOS”，但递推实际读取的是 correction 后的 `response_mask`。因此 Tool Agent Loop 的工具 observation 和 rollout rejection 置零的 assistant token 都会触发 reset。先看只有 tool observation 的例子：

```text
内容:          assistant  tool-observation  final-assistant
response_mask:     1             0                 1
reward:            0             0                 1
```

`gamma=1` 时当前代码得到的 raw returns 是：

```text
returns = [0, 1, 1]
```

中间 observation 的 return 会被 `response_mask` 忽略，而工具调用之前的 assistant token 接收不到最终 reward。也就是说，这个 estimator 当前不会像 GAE 那样跨过 mask 0 传播 return。若 mask 0 来自 rejection，rejected token 自身不参与白化/loss，而且它还会阻断更早 token 接收其右侧 reward。对多轮 tool-agent 或启用 rollout rejection 的训练选择它之前，应先确认这种 credit-assignment 语义正是你想要的。

### 9.2 `reinforce_plus_plus_baseline`

baseline 变体先减去同一 `uid` group 的**包含自身**的均值，再把标量广播到 token，最后在全 batch 上 whiten。实现见
[`compute_reinforce_plus_plus_baseline_outcome_advantage`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L533-L584)。

这里有一个 singleton 特判：若某个 `uid` group 只有一条回答，当前实现把该组 baseline 设为 0，所以白化前保留原始轨迹 score，而不是用“自身均值”把它减成 0。随后该标量仍会广播到 post-correction mask 保留的 token，并与全 batch 一起白化；因此这个特判既不保证最终 advantage 非零，也不能避免“全 batch 只有一个有效 token”时的样本方差错误。它的轨迹 score 仍在乘 mask 之前求和，所以同样受 7.3 节说明的 rejection 时序影响。

与 RLOO 一样，当前 V1 不会为这个 baseline 变体折叠 custom Agent Loop 的 multi-output session；每个 output row 都会进入 `uid` group，具体语义见 7.5 节。

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

只会选择本节的 reward-to-go/whitening，不会自动把单个有效 token 的 loss 改成 $-\log\pi_\theta(a_t\mid s_t)\,A_t$。这个式子表示“负的 token log-policy 乘 advantage”：负号让最小化 loss 等价于提高正 advantage 动作的概率；$\log$ 是自然对数；$\pi_\theta(a_t\mid s_t)$ 是参数为 $\theta$ 的策略在状态 $s_t$ 下选择动作 $a_t$ 的概率；$A_t$ 是该动作的 advantage，相邻两项表示相乘。实际 batch loss 还要只对 post-correction mask 为 1 的 token 聚合。这一点在下一章展开。

---

## 10. 其他 estimator 的定位

注册表位于
[`AdvantageEstimator`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L88-L110)。初学阶段先掌握 GAE、GRPO、RLOO、REINFORCE++，再阅读下表中的扩展。

| estimator | baseline/变换 | 适用信号 | 入口 |
|---|---|---|---|
| `grpo_vectorized` | 默认 `norm_adv_by_std_in_grpo=True` 时与 GRPO 数学相同；当前 V1 dispatcher 不转发该 flag，因此配置 `false` 对它无效 | outcome | [`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L334-L358) |
| `rloo_vectorized` | group size 大于 1 时与 RLOO 相同；singleton advantage 归零 | outcome | [`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L831-L866) |
| `remax` | sampled return 减 greedy rollout reward | outcome | [`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L732-L765) |
| `grpo_passk` | group 至少需 2 条；仅最佳样本得到 `max - second_max`，默认还除以 group 样本标准差 `std + epsilon` | Pass@k | [`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L471-L530) |
| `opo` | 按 post-correction `response_mask.sum(-1)`，即未被 rejection 去掉的 assistant 动作 token 数加权 group reward baseline；不计 tool observation | outcome/长度校正 | [`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L639-L690) |
| `gpg` | group-centering 后乘 `B / max(count_nonzero(scores), 1)`，再除以 `f_norm`；即按非零轨迹比例的倒数缩放 | outcome | [`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L768-L828) |
| `gdpo` | 算法函数会对每个 reward 维度分别做 GRPO，再加权和并 whiten；当前默认 V1 数据路径未传递所需字段，见 3.2 节 | 多维 outcome reward | [`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L361-L468) |
| `optimal_token_baseline` | 用 token/path 方差 proxy 学构造更细 baseline | 单轮高级用法 | [`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L869-L987) |
| `tir_optimal_token_baseline` | 面向多轮轨迹的 optimal-token 版本 | 多轮高级用法 | [`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L988-L1118) |

这些函数被注册并不等于每个 trainer/backend 组合都具备相同的端到端辅助流程。例如 ReMax 还需要额外生成 greedy baseline；阅读高级 estimator 时必须继续检查 trainer 的数据准备路径，而不能只看到 enum 就认为已经完整接线。

---

## 11. 四种核心 estimator 对比

| 特性 | GAE | GRPO | RLOO | REINFORCE++ |
|---|---|---|---|---|
| 需要 critic | 是 | 否（默认关闭，可显式开启） | 否（默认关闭，可显式开启） | 否（默认关闭，可显式开启） |
| 需要同 prompt 多采样 | 否 | 是 | 是 | 否 |
| baseline | $V(s_t)$（critic 对时刻 $t$ 的状态 $s_t$ 所预测的价值；$V$ 是 value 函数） | group mean | leave-one-out mean | 无显式 baseline；全 batch whitening |
| token credit | 逐 token 递推 | 轨迹标量广播 | 轨迹标量广播 | reward-to-go |
| 当前可跨 tool observation 传播最终 reward | 是，跳过 mask 0 | 是，整轨迹求和后广播 | 是，整轨迹求和后广播 | 否，mask 0 会 reset |
| `returns` 的含义 | critic target | 等于 advantage 的接口字段 | 等于 advantage 的接口字段 | 未 whiten 的 reward-to-go |

表中的 mask 都是优势估计时的 post-correction `response_mask`：tool observation、padding 和 rejected assistant token 都为 0。GAE 会跨过这三类 0；GRPO/RLOO 先对未清零的 reward 求轨迹和，再只向 mask 为 1 的位置广播；REINFORCE++ 则在每个 0 处 reset。

表中“同 prompt 多采样”默认每个 rollout session 只产生一个 output。对自定义 multi-output Agent Loop，只有精确的 `grpo` estimator 应用 session-final 折叠；RLOO 和其他 group estimator 的实际分组行数可能大于 `rollout.n`，见 7.5 节。

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

假设这里的 Agent Loop 每个 session 只返回一个 output，一个 trainer step 会围绕两个 prompt 形成两个 group：

```text
uid=A -> trajectory A0, A1, A2 -> rewards [1.0, 0.0, 0.5]
uid=B -> trajectory B0, B1, B2 -> rewards [0.0, 0.0, 0.0]
```

优势：

```text
A group -> [ 0.999998, -0.999998, 0.0]  # default epsilon=1e-6
B group -> [ 0.0,  0.0, 0.0]
```

然后每个标量被广播到各自 post-correction `response_mask == 1` 的 assistant token；tool observation、padding 和 rejected assistant 位置归零。下一章的 PPO loss 再用这些 token-level advantage 更新 actor。

---

## 13. 调试 reward/advantage 的检查顺序

遇到“reward 明明正常，但模型不学习”时，按以下顺序看：

1. `rm_scores.sum(-1)` 是否等于你期望的轨迹总分。
2. `token_level_rewards.sum(-1)` 是否因 KL 系数过大而改变符号或量级。
3. 分阶段检查 mask：Agent Loop 写入时 `response_mask` 是否正确区分 assistant 与 tool observation；rollout correction 后是否只额外置零了预期的 rejected assistant token；`loss_mask` 是否仍保留原始 assistant-token 计数。
4. 先区分 Agent Loop 是单 output 还是 multi-output：单 output 时，同一 `uid` group 通常应有 `rollout.n` 行；multi-output 时，对 `grpo` 应按 `{uid}_{session_id}` 检查 `rollout.n` 个 session-final outputs，对 RLOO 等其他 group estimator 则必须检查实际 `{uid}_{session_id}_{index}` output rows，行数可能大于 `rollout.n`。
5. GRPO/RLOO 的 group 是否全部同分；同分就没有相对信号。
6. `advantages[response_mask.bool()]` 的均值、标准差、正负比例是否合理。
7. 多轮工具训练或启用 rollout rejection 时若使用 REINFORCE++，检查 observation/rejected-token 的 mask 0 是否意外截断最终 reward。
8. 最后再看 PPO ratio、clip fraction 和 loss；advantage 生成之前的问题，不能靠调 clip ratio 修复。

下一章会把 `advantages` 接到 actor/critic 的实际 loss、mini-batch 和 micro-batch 更新路径上。
