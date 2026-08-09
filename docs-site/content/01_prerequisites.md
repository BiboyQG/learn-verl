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

设词表大小为 `V`，一条长度为 `L` 的序列通常有：

- `input_ids`: `[L]`
- batch 后的 `input_ids`: `[B, L]`
- 模型输出 logits: `[B, L, V]`

chat template 还会插入角色标记、消息边界和工具 schema。对于当前 V1 Agent Loop，dataset 主要提供 `raw_prompt`；真正的 chat-template/tokenization 位于 rollout/Agent Loop 一侧，而不是假定 dataset 已经给出最终 `input_ids`。

### Logits、概率与 log-prob

模型在位置 `t` 输出对下一 token 的 logits：

$$
z_t \in \mathbb{R}^{V}
$$

softmax 后得到概率：

$$
\pi_\theta(a_t\mid s_t)
= \frac{\exp z_{t,a_t}}{\sum_{v=1}^{V}\exp z_{t,v}}
$$

RL 训练通常使用 log-prob：

$$
\log \pi_\theta(a_t\mid s_t)
$$

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

这里至少要区分四个 policy 概念：

- **rollout/behavior policy**：推理 backend 真正采样 trajectory 时的策略，记录为 `rollout_log_probs`；
- **old/proximal policy**：本批 actor update 的固定锚点，记录为 `old_log_probs`；
- **reference policy**：通常是冻结的 SFT 模型，用于 KL 约束；
- **current policy**：正在反向传播的 actor。

理想 on-policy PPO 中 rollout 与 old 是同一个策略。当前 V1 默认 decoupled 路径会让 training actor 在更新前重算 `old_log_probs`；名义权重可能相同，但它仍不等于 rollout backend 当时记录的行为概率。只有 bypass 路径才直接令 `old_log_probs = rollout_log_probs`。

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

- 状态 `s_t`
- 动作 `a_t`
- 状态转移 `P(s_{t+1}|s_t,a_t)`
- 奖励 `r_t`
- 策略 `π(a_t|s_t)`

映射到 LLM：

| RL | LLM / Agent |
|---|---|
| 状态 `s_t` | prompt 加上目前为止的 token、工具 observation |
| 动作 `a_t` | 模型选择的下一个 token，或更高层的 tool call |
| 策略 `πθ` | actor LLM |
| 环境 | tokenizer 规则、tool、sandbox、用户模拟器等 |
| trajectory | 一次完整回答或多轮交互 |
| reward | 正确性、格式、偏好、安全性等评分 |

严格来说，语言生成中的状态是完整历史；因为这个历史可见，token-level 展开仍可按 MDP 形式处理。

### Return 与 Advantage

从时刻 `t` 开始的折扣回报：

$$
G_t = \sum_{k=0}^{T-t-1}\gamma^k r_{t+k}
$$

value function 预测从当前状态出发的期望回报：

$$
V^\pi(s_t)=\mathbb{E}[G_t\mid s_t]
$$

advantage 衡量某个实际动作比“当前状态下的平均水平”好多少：

$$
A_t = Q^\pi(s_t,a_t)-V^\pi(s_t)
$$

直觉上：

- `A_t > 0`：提高这个 token/action 的概率；
- `A_t < 0`：降低它的概率；
- `A_t ≈ 0`：这个动作没有明显优劣信号。

### Policy Gradient

最基础的策略梯度目标具有如下形态：

$$
\nabla_\theta J(\theta)
\approx
\mathbb{E}_t\left[
A_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right]
$$

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

其中：

```text
r_t = exp(current_log_prob - old_log_prob)
```

当更新把 token 概率推得过远时，clip 会截住该样本继续改善 surrogate objective 的激励。它不是对参数或最终 ratio 的硬投影，不能保证训练后所有 ratio 都落在 `[1-ε,1+ε]`。实现里通常最小化负目标，因此代码中的 `pg_loss` 符号可能与论文最大化目标相反；当前默认 vanilla loss 对负 advantage 还支持 dual-clip 扩展，后文会单独解释。

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

Generalized Advantage Estimation：

$$
A_t^{\text{GAE}}
= \delta_t
+ (\gamma\lambda)\delta_{t+1}
+ (\gamma\lambda)^2\delta_{t+2}
+ \cdots
$$

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
    advantage[t] = last_gae
return_ = advantage + value
```

这段仍是简化伪代码。Tool Agent 的 observation token 具有 `response_mask=0`，但不代表 episode 在那里终止；当前实现跳过 observation 的 value/TD-error，同时保留 `next_value` 和 `last_gae`，从而连接 observation 两侧相邻的模型 action。下游 loss 仍只消费 mask 有效的位置。

## 1.5 GRPO：用组内相对表现代替 learned critic

GRPO 常对同一个 prompt 采样多条 response，然后在组内标准化 reward。简化形式：

$$
A_i = \frac{R_i-\operatorname{mean}(R_{g(i)})}
{\operatorname{std}(R_{g(i)})+\varepsilon}
$$

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

在 sampled token 上会使用不同 estimator。框架里常见两种放置方式：

1. **KL in reward**：从 token reward 中减去 KL penalty，再估 advantage；
2. **KL loss**：policy loss 之外另加 KL 项。

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

这就是 sample staleness。标准 PPO 在 old policy 就是 behavior policy 时，ratio 能容纳有限的策略差异；但当前 decoupled 路径中的 `exp(current-old)` 主要约束 proximal update，并不会自动校正 `rollout policy → old policy` 的偏差。后者要靠模型版本阈值、rollout correction/rejection，或明确采用 old=rollout 的 bypass 语义。verl 的同步、colocate async、separate async 模式，本质上是在吞吐、显存复用和 staleness 之间取舍。

## 1.8 分布式训练最小知识

### Rank 与 Process Group

分布式程序通常启动多个进程，每个进程有一个全局 `rank`。一组进程通过 process group 执行 collective：

- all-reduce：聚合梯度/统计；
- all-gather：收集分片；
- reduce-scatter：规约后再分片；
- broadcast：从一个 rank 发给其他 rank。

### Data Parallel

每个 rank 处理不同数据。朴素 DP 每张 GPU 都持有完整参数；FSDP/ZeRO 则把参数、梯度和 optimizer state 分片，降低单卡显存。

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
@ray.remote(num_gpus=1)
class Worker:
    def compute(self, x):
        return x * 2

w = Worker.remote()
future = w.compute.remote(21)
result = ray.get(future)
```

在 verl 中：

- driver/controller 决定调用顺序；
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

1. `old_log_prob` 与 `ref_log_prob` 为什么不能互换？
2. 工具 observation 为什么 `attention_mask=1`，但 `response_mask=0`？
3. GRPO 的 group 是按 batch 随意分组，还是按原始 prompt 分组？
4. Ray placement 与 FSDP collective 各自负责什么？
5. 为什么 training engine 更新后，rollout server 不一定自动得到新参数？

这些概念会在后续章节与具体 class、字段和调用链一一对应。
