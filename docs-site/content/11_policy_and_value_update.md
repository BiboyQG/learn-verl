# 11. Policy 与 Value Update：PPO loss 怎样真正更新参数

上一章已经得到：

```text
advantages: [B, response_length]
returns:    [B, response_length]
```

这一章继续追踪它们：

```text
advantages -> policy loss -> actor backward -> optimizer.step()
returns    -> value loss  -> critic backward -> optimizer.step()
```

重点不是背一条 PPO 公式，而是把公式中的每一个符号映射到 verl 的字段、配置和 worker 调用。

---

## 1. 当前 V1 trainer 的一次更新全景

[`PPOTrainer._step_once`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L536-L586) 把一次本地更新明确分成九步：

```text
1. 从 replay buffer 取一批 rollout
2. 计算 reward（若未在异步 reward loop 中计算）
3. 按 token workload 平衡到各 DP rank
4. 计算 old_log_probs
5. 可选：计算 ref_log_prob
6. 可选：critic 计算 values
7. reward shaping + advantage/return
8. 可选：更新 critic
9. 更新 actor
```

源码骨架可以简化成：

```python
batch = replay_buffer.sample(...)
batch = compute_reward(batch)
batch = balance_batch(batch)
batch = compute_old_log_prob(batch)

if use_reference_policy:
    batch = compute_ref_log_prob(batch)
if use_critic:
    batch = compute_values(batch)

batch = compute_advantage(batch)

if use_critic:
    update_critic(batch)
update_actor(batch)
```

这里有一个非常重要的架构特点：**rollout、advantage estimator、policy loss 是三个可独立组合的层次**。

- Rollout 决定数据从哪一个 policy/engine 采样。
- `algorithm.adv_estimator` 决定怎样得到 `advantages`。
- `actor_rollout_ref.actor.policy_loss.loss_mode` 决定怎样用这些 advantage 更新 actor。

所以 `adv_estimator=grpo` 并不代表“整个训练器换成另一套 GRPO trainer”；它仍复用 PPO 风格的 actor update、mini-batch、ratio 和 clipping 基础设施。

---

## 2. 先认清四个 policy

在大模型 post-training 中，“当前模型”和“旧模型”往往不止两个。建议用下面四个名字思考：

| 数学名称 | verl 字段 | 作用 | 何时变化 |
|---|---|---|---|
| $\pi_{rollout}$ | `rollout_log_probs` | 真正生成 trajectory 的 rollout engine policy | rollout 权重同步后变化 |
| $\pi_{old}$ | `old_log_probs` | PPO ratio 的固定锚点 | 每批 actor update 前重算一次 |
| $\pi_\theta$ | actor forward 得到的 `log_prob` | 正在反向传播、不断更新的 policy | 每个 optimizer step 都变化 |
| $\pi_{ref}$ | `ref_log_prob` | 冻结 reference policy，用于 KL regularization | 通常不变 |

### 2.1 默认是“三策略 decoupled”语义

当前默认 rollout correction 配置的 `bypass_mode: false`。V1 trainer 会在 actor 更新之前用训练模型重新计算一次 `old_log_probs`：

```python
output = actor_rollout_wg.compute_log_prob(batch)
data["old_log_probs"] = response_from_nested(
    data.pop("log_probs"), data["response_mask"]
)
```

见
[`trainer_base.py::_compute_old_log_prob`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L1479-L1538)。这时：

```text
pi_rollout --负责生成
pi_old     --负责 PPO proximal anchor
pi_theta   --负责当前训练 forward/backward
```

同步训练中三者通常很接近，但仍可能因 rollout engine 数值实现、权重版本或异步延迟而不同。

### 2.2 bypass mode

若 `algorithm.rollout_correction.bypass_mode=true`，trainer 直接令：

```python
old_log_probs = rollout_log_probs
```

即只有 $\pi_{rollout}$ 与 $\pi_\theta$ 两个动态 policy。对应逻辑见
[`trainer_base.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L1480-L1493)。bypass mode 还可以在 PPO-clip 和显式 REINFORCE+importance-sampling loss 之间选择；这是高级 off-policy 主题，不是普通 PPO 必需配置。

---

## 3. 从 log-probability 到 PPO ratio

对第 $t$ 个已采样 token $a_t$：

$$
\log p_{old,t}=\log\pi_{old}(a_t\mid s_t),
\qquad
\log p_{\theta,t}=\log\pi_\theta(a_t\mid s_t).
$$

PPO 的 importance ratio 是：

$$
\rho_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}
=\exp(\log p_{\theta,t}-\log p_{old,t}).
$$

源码对应：

```python
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
ratio = torch.exp(negative_approx_kl)
```

见
[`compute_policy_loss_vanilla`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L1283-L1374)。clamp 只是防止 `exp` 数值爆炸。

### 3.1 数值直觉

若某个 token：

```text
old_log_prob = -1.2
log_prob     = -1.0
```

则：

$$
\rho=e^{-1.0-(-1.2)}=e^{0.2}\approx1.221.
$$

这表示当前 policy 给这个已采样 token 的概率，大约是旧 policy 的 1.221 倍。

- `ratio == 1`：policy 没变。
- `ratio > 1`：该 token 概率提高。
- `ratio < 1`：该 token 概率降低。

---

## 4. PPO clipped objective

### 4.1 标准公式

PPO 想最大化：

$$
L^{clip}_t(\theta)
=\min\left(
\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right).
$$

verl 的优化器执行最小化，因此代码保存的是负号后的 loss：

```python
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(
    ratio, 1 - clip_ratio_low, 1 + clip_ratio_high
)
pg_losses = torch.maximum(pg_losses1, pg_losses2)
```

“最大化两个 objective 的较小者”等价于“最小化两个负 loss 的较大者”。

### 4.2 正 advantage 的数值例子

假设：

```text
A       = +1
ratio   = 1.221
epsilon = 0.2
```

未裁剪与裁剪 objective：

```text
ratio * A       = 1.221
clip(ratio) * A = 1.200
PPO objective   = min(1.221, 1.200) = 1.200
loss            = -1.200
```

这个好动作的概率已经提高超过 20%，PPO 不再奖励超出部分。

### 4.3 负 advantage 的数值例子

保持相同 ratio，但：

```text
A = -1
```

则：

```text
ratio * A       = -1.221
clip(ratio) * A = -1.200
PPO objective   = min(-1.221, -1.200) = -1.221
loss            = 1.221
```

旧 policy 已经认为这个动作不好，而当前 policy 反而提高了它的概率；PPO 会继续强烈惩罚。clip 不是“ratio 超出区间就一律截断”，它只截断会让 surrogate objective 继续变好的方向。

### 4.4 asymmetric clip 与 dual clip

verl 可以分别设置：

```yaml
actor_rollout_ref:
  actor:
    clip_ratio: 0.2
    clip_ratio_low: 0.2
    clip_ratio_high: 0.2
    clip_ratio_c: 3.0
```

- `clip_ratio_low` 控制下界 $1-\epsilon_{low}$。
- `clip_ratio_high` 控制上界 $1+\epsilon_{high}$。
- `clip_ratio_c=C>1` 对负 advantage 再加入 dual-clip 下界。

当前 dual-clip 最大化的形式可写成：

$$
L_t^{dual}=
\begin{cases}
L_t^{clip}, & A_t\ge0,\\
\max(L_t^{clip}, C A_t), & A_t<0.
\end{cases}
$$

它防止负 advantage 与极大 ratio 相乘时产生异常大的梯度。对应分支在
[`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L1353-L1359)。

---

## 5. actor loss 的真实组装过程

公共 actor loss 入口是
[`workers/utils/losses.py::ppo_loss`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/utils/losses.py#L57-L144)。它做四件事：

1. 把 no-padding/nested model output 重新对齐成 `[B, response_length]`。
2. 读取 `old_log_probs`、`advantages`、`response_mask`。
3. 根据 `policy_loss.loss_mode` dispatch 到具体 policy loss。
4. 可选加入 entropy bonus 和 reference KL loss。

简化后：

```python
policy_loss_fn = get_policy_loss_fn(config.policy_loss.loss_mode)
pg_loss, metrics = policy_loss_fn(
    old_log_prob=old_log_probs,
    log_prob=current_log_probs,
    advantages=advantages,
    response_mask=response_mask,
    config=config,
)

loss = pg_loss
loss -= entropy_coeff * entropy

if use_kl_loss:
    loss += kl_loss_coef * reference_kl
```

也就是：

$$
\mathcal L_{actor}
=\mathcal L_{PG}
-c_H\,H(\pi_\theta)
+c_{KL}\,\widehat D_{KL}(\pi_\theta,\pi_{ref}).
$$

### 5.1 `adv_estimator` 与 `loss_mode` 是正交的

默认 actor 配置是：

```yaml
policy_loss:
  loss_mode: vanilla
```

见
[`trainer/config/actor/actor.yaml`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/config/actor/actor.yaml#L54-L76)。因此以下配置：

```yaml
algorithm:
  adv_estimator: reinforce_plus_plus
```

会使用 REINFORCE++ 计算 advantage，但 policy loss 仍是 PPO clipped loss。

如果真要用无 clipping 的 $-A\log\pi$ loss，当前专门实现是
[`compute_policy_loss_reinforce`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L2330-L2407)，它由 `bypass_mode` 的 `loss_type=reinforce` 分支调用，见
[`compute_policy_loss_bypass_mode`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L2410-L2546)。不要仅凭 estimator 名字推断 loss。

### 5.2 其他 policy loss

`core_algos.py` 还注册了 `dppo_tv`、`dppo_kl`、`gspo`、`sapo`、`gpg`、`clip-cov`、`kl-cov`、`geo_mean`、`dro`、`cispo` 等 loss。它们共享相同的 dispatch 接口：

```python
(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, config, ...)
```

注册与查找逻辑位于
[`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L37-L85)。初学时先把 `vanilla` 理解透，再读这些扩展；否则很容易把“advantage 算法”和“ratio clipping 算法”混在一起。

---

## 6. Entropy bonus：鼓励保留探索

token-level categorical entropy 是：

$$
H_t(\pi_\theta)
=-\sum_a\pi_\theta(a\mid s_t)\log\pi_\theta(a\mid s_t).
$$

entropy 高，说明概率分布更平；entropy 低，说明模型几乎确定只选少数 token。

verl 最小化：

$$
\mathcal L_{actor}=\mathcal L_{PG}-c_H H.
$$

所以 `entropy_coeff > 0` 会奖励更高 entropy。配置是：

```yaml
actor_rollout_ref:
  actor:
    entropy_coeff: 0.0
    calculate_entropy: false
```

如果 `entropy_coeff != 0`，trainer 会自动请求 actor forward 计算 entropy，见
[`trainer_base.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L1672-L1703)。entropy 同样只在 `response_mask` 选中的 assistant token 上聚合。

---

## 7. 两条完全不同的 reference KL 路径

KL regularization 的目的，是不让 post-trained policy 偏离冻结 reference policy 太远。verl 提供两条路径。

| 对比 | reward-side KL | loss-side KL |
|---|---|---|
| 开关 | `algorithm.use_kl_in_reward` | `actor_rollout_ref.actor.use_kl_loss` |
| 使用的 policy | `old_log_probs` vs `ref_log_prob` | 当前 `log_prob` vs `ref_log_prob` |
| 发生时机 | advantage 计算前 | 每次 actor forward/backward |
| 数学形式 | $r_t=s_t-\beta KL_t$ | $\mathcal L+=c_{KL}KL_t$ |
| 会不会改变 `token_level_rewards` | 会 | 不会 |
| 系数 | fixed/adaptive KL controller | 固定 `kl_loss_coef` |
| 会不会间接改变 advantage | 会 | 不会，直接改变梯度 |

### 7.1 reward-side KL

实现位于
[`apply_kl_penalty`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/ray_trainer.py#L78-L117)：

$$
r_t=s_t-\beta\widehat D_{KL,t}.
$$

若 controller 为 adaptive，系数更新为：

$$
e=\operatorname{clip}\left(\frac{KL_{current}}{KL_{target}}-1,-0.2,0.2\right),
$$

$$
\beta\leftarrow\beta
\left(1+e\frac{n_{steps}}{horizon}\right).
$$

对应源码在
[`AdaptiveKLController`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L153-L175)。KL 高于 target 时 $\beta$ 增大，低于 target 时减小。

### 7.2 loss-side KL

actor forward 得到当前 `log_prob` 后：

```python
kld = kl_penalty(log_prob, ref_log_prob, config.kl_loss_type)
kl_loss = agg_loss(kld, response_mask, config.loss_agg_mode, ...)
policy_loss += config.kl_loss_coef * kl_loss
```

见
[`workers/utils/losses.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/utils/losses.py#L131-L143)。因为 $\pi_\theta$ 会在每个 mini-batch optimizer step 后改变，所以 loss-side KL 会持续追踪最新 actor。

### 7.3 KL estimator 的精确形式

令：

$$
d=\log\pi(a\mid s)-\log\pi_{ref}(a\mid s).
$$

[`kl_penalty_forward`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L2213-L2248) 支持：

| 配置 | token estimator |
|---|---|
| `kl` / `k1` | $d$ |
| `abs` | $\lvert d\rvert$ |
| `mse` / `k2` | $\tfrac12 d^2$ |
| `low_var_kl` / `k3` | $e^{-d}+d-1$ |
| `full` | 当前函数中未实现 |

`k1` 单个样本可以为负；它是在对应采样分布下取期望后才成为 KL。`abs`、`k2`、`k3` 提供不同的偏差、方差与非负性权衡。`k1+`、`k3+` 等带 `+` 形式会用 straight-through 技巧，让前向值与反向梯度采用不同 estimator，见
[`kl_penalty`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L2185-L2210)。

### 7.4 两条 KL 可以同时开，但通常要有意为之

只要任一开关启用，trainer 就会创建/使用 reference policy，见
[`need_reference_policy`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/utils.py#L75-L80)。两条路径同时开启不会被禁止，但配置校验会打印 notice，见
[`utils/config.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/utils/config.py#L169-L170)。

同时启用等于既改变 reward/advantage，又在 loss 上再施加一次 reference regularization。除非你明确设计了这个目标，否则很容易把 KL 约束加重两次。

### 7.5 `actor/ppo_kl` 不是 reference KL

vanilla loss 里的指标：

```python
ppo_kl = masked_mean(old_log_prob - log_prob, response_mask)
```

测的是当前 policy 与 **PPO old policy** 的采样近似差异，用来观察一次 PPO update 移动了多远。它不是 $\pi_\theta$ 与 $\pi_{ref}$ 的 KL。变量都叫 KL，但回答的是两个不同问题。

---

## 8. critic 的 clipped value loss

GAE 需要 critic。对每个有效 token：

- `values`：这批数据进入 PPO 更新前的旧 value prediction。
- `vpreds`：当前 critic forward 的新 prediction。
- `returns`：上一章 GAE 生成的 target。

先裁剪 critic 的变化：

$$
V^{clip}_t
=\operatorname{clip}(V^{new}_t,
V^{old}_t-\epsilon_v,
V^{old}_t+\epsilon_v).
$$

再取两个平方误差中更大的一个：

$$
\mathcal L_V
=\frac12\,\mathbb E_t\left[
\max\left(
(V^{new}_t-G_t)^2,
(V^{clip}_t-G_t)^2
\right)
\right].
$$

实现见
[`compute_value_loss`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L2122-L2182)，worker 入口见
[`workers/utils/losses.py::value_loss`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/utils/losses.py#L147-L205)。

### 8.1 数值例子

假设：

```text
old value = 0.4
new value = 0.9
return    = 1.0
cliprange = 0.2
```

那么：

```text
clipped new value = 0.6
unclipped error^2 = (0.9 - 1.0)^2 = 0.01
clipped error^2   = (0.6 - 1.0)^2 = 0.16
value loss        = 0.5 * max(0.01, 0.16) = 0.08
```

虽然 `new value=0.9` 很接近 target，但它一次从 0.4 跳得太远，不能靠“刚好更接近 target”逃过惩罚。这与 PPO policy clipping 的保守更新思想一致。

critic 配置有自己的：

```yaml
critic:
  ppo_mini_batch_size: ...
  ppo_micro_batch_size_per_gpu: ...
  ppo_epochs: ...
  cliprange_value: ...
```

actor 和 critic 可以有不同的 mini/micro batch 与 epoch 数。

---

## 9. `response_mask` 最终怎样进入 loss

所有关键 loss 都遵守同一规则：

```python
valid_loss = aggregate(loss_matrix, response_mask)
```

因此：

- prompt token 不在 `[B,response_length]` loss matrix 中。
- tool observation 的 `response_mask=0`，不产生 policy/value/KL/entropy loss。
- response padding 的 mask 也是 0。
- V1 为批次整除而添加的 synthetic padding sample，其整个 `response_mask` 都是 0，因此不会产生梯度，见
  [`padding_utils.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/padding_utils.py#L70-L138)。

模型输出与 response token 的对齐还需要左移一位：位置 $t$ 的 logits 预测下一个 token。no-padding 输出还原为 response tensor 的切片逻辑在
[`workers/utils/padding.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/utils/padding.py#L99-L143) 和
[`response_from_nested`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/utils/padding.py#L196-L212)。

---

## 10. Loss aggregation：谁在平均谁

设第 $i$ 条轨迹有效动作 token 数为 $T_i$，token loss 为 $\ell_{i,t}$，mask 为 $m_{i,t}$。`loss_agg_mode` 会实质改变不同长度轨迹的相对权重。

实现统一在
[`agg_loss`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L1138-L1204)。

### 10.1 `token-mean`

$$
\mathcal L
=\frac{\sum_i\sum_t m_{i,t}\ell_{i,t}}
{\sum_i\sum_t m_{i,t}}.
$$

每个有效 token 权重相等。长回答包含更多 token，因此整条轨迹的总影响通常更大。当前 actor 默认是这个模式。

### 10.2 `token-sum`

$$
\mathcal L=\sum_i\sum_t m_{i,t}\ell_{i,t}.
$$

不做平均，梯度大小会随有效 token 总数增长；通常只有在外部已经精确控制 scale 时使用。

### 10.3 `seq-mean-token-sum`

$$
\mathcal L
=\frac1B\sum_i\sum_t m_{i,t}\ell_{i,t}.
$$

先对每条轨迹 token 求和，再对轨迹平均。长轨迹仍有更大总权重。

### 10.4 `seq-mean-token-mean`

$$
\mathcal L
=\frac1B\sum_i
\frac{\sum_t m_{i,t}\ell_{i,t}}
{\sum_t m_{i,t}}.
$$

每条轨迹先做 token mean，所以每条非空轨迹大致等权。这更接近原始 GRPO 的 sample-level loss，但在长 CoT 中可能改变优化稳定性。

### 10.5 `seq-mean-token-sum-norm`

$$
\mathcal L
=\frac1B\sum_i
\frac{\sum_t m_{i,t}\ell_{i,t}}{C}.
$$

$C$ 来自 `actor.loss_scale_factor`；若为 `null`，当前实现使用 loss matrix 的 horizon。把 $C$ 设成固定 `max_response_length` 是 Dr.GRPO 常见设置，可避免 batch 间 normalization factor 漂移。现有说明见
[`docs/algo/grpo.md`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/docs/algo/grpo.md#L51-L62)。

### 10.6 一个长度例子

两条轨迹 token loss 都为 1：

```text
trajectory A: 2 个有效 token -> [1, 1]
trajectory B: 6 个有效 token -> [1, 1, 1, 1, 1, 1]
```

- `token-mean`：A 占总梯度权重的 $2/8$，B 占 $6/8$。
- `seq-mean-token-sum`：A 的序列 loss 为 2，B 为 6，再求平均；仍是 1:3。
- `seq-mean-token-mean`：A、B 的序列 loss 都为 1；两条轨迹等权。

所以调 `loss_agg_mode` 不是纯性能优化，而是改变训练目标的权重结构。

---

## 11. Batch、mini-batch、micro-batch 到底是什么

这三个词最容易造成配置错误。

### 11.1 rollout/train batch

`data.train_batch_size=B` 是每个 trainer step 的原始 prompt group 数。若每个 prompt 采样：

```yaml
actor_rollout_ref:
  rollout:
    n: n
```

标准“每个 session 一个 output”的路径会产生：

$$
N_{trajectory}=B\times n.
$$

例如 `train_batch_size=128, rollout.n=8`，会有 1024 条 trajectory。

### 11.2 PPO mini-batch

mini-batch 是**一次 optimizer step 的有效优化批次**。一个 rollout batch 会被切成多个 mini-batch，并可通过 `ppo_epochs` 重复使用。

当前 trainer 会把配置中的：

```yaml
actor_rollout_ref:
  actor:
    ppo_mini_batch_size: M
```

乘以 `rollout.n` 后，再传给训练 worker：

```python
ppo_mini_batch_size = actor.ppo_mini_batch_size
ppo_mini_batch_size *= rollout.n
extra_info["mini_batch_size"] = ppo_mini_batch_size
```

见
[`trainer_base.py::_update_actor`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L1672-L1705)。critic 路径也做同样乘法，见
[`trainer_base.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L1649-L1669)。

因此在当前标准路径中，配置 `M=32, n=8` 对应 worker 中每个 actor mini-batch 为 256 条 trajectory。可以把配置值理解成 **prompt-group 等价大小**；源码中的实际 trajectory mini-batch 是 $M\times n$。

### 11.3 micro-batch

mini-batch 可能无法一次放进 GPU，于是每个 DP rank 再把本地 mini-batch 切成多个 micro-batch：

```yaml
actor_rollout_ref:
  actor:
    ppo_micro_batch_size_per_gpu: m
```

每个 micro-batch 做一次 forward/backward，但先只累积 gradient；处理完完整 mini-batch 后才 `optimizer.step()`。

engine 的精确顺序是：

```python
optimizer_zero_grad()
forward_backward_batch(...)  # 内部遍历 micro-batches 并累积 gradient
optimizer_step()
```

见
[`workers/engine/base.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/engine/base.py#L113-L130)。FSDP micro-batch 循环与最后一次 gradient sync 在
[`workers/engine/fsdp/transformer_impl.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/engine/fsdp/transformer_impl.py#L700-L748)。

所以：

- mini-batch 改变一次 optimizer update 看多少训练样本，可能改变优化行为。
- micro-batch 主要是显存/吞吐旋钮；在正确归一化下，不应改变有效 batch objective。

### 11.4 一个从头算到尾的例子

配置：

```text
train_batch_size                     B = 128 prompts
rollout.n                            n = 8
actor.ppo_mini_batch_size            M = 32 prompt-equivalents
data-parallel size                   D = 8 GPUs
actor.ppo_micro_batch_size_per_gpu   m = 2 trajectories/GPU
actor.ppo_epochs                         2
```

逐层展开：

```text
rollout batch:
    128 * 8 = 1024 trajectories

实际 global mini-batch:
    32 * 8 = 256 trajectories

每个 DP rank 的 local mini-batch:
    256 / 8 = 32 trajectories

每次 global micro-batch wave:
    2 * 8 = 16 trajectories

每个 mini-batch 的 gradient accumulation wave 数:
    256 / 16 = 16

每个 epoch 的 optimizer step 数:
    1024 / 256 = 4

ppo_epochs=2 的总 actor optimizer step 数:
    4 * 2 = 8
```

在每个 mini-batch 内，`old_log_probs` 保持不变；但每次 optimizer step 后 $\pi_\theta$ 都改变，所以后续 mini-batch/epoch 重新 forward 得到的 current `log_prob` 会变化。

### 11.5 dynamic batch size

若：

```yaml
actor_rollout_ref:
  actor:
    use_dynamic_bsz: true
    ppo_max_token_len_per_gpu: 24576
```

micro-batch 不再固定包含 $m$ 条 trajectory，而是根据 token workload，在每个 GPU 的 token budget 内动态打包。实现入口是
[`prepare_micro_batches`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/engine/utils.py#L83-L121)：

- `use_dynamic_bsz=false`：按 `micro_batch_size_per_gpu` 等数量切分。
- `use_dynamic_bsz=true`：按 `max_token_len_per_gpu` 和 sequence-parallel size 重排。

dynamic batching 改变的是 micro-batch 形状和数量，不改变上层 mini-batch 应代表的全局 objective。

### 11.6 log-prob/value inference 有独立 micro-batch

不要把训练 micro-batch 与这些 forward-only 配置混在一起：

- `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`：重算 actor old log-prob。
- `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`：计算 reference log-prob。
- `critic.forward_micro_batch_size_per_gpu`：critic value inference。
- 相应的 `*_max_token_len_per_gpu`：dynamic forward token budget。

它们主要影响推理阶段显存和吞吐，不等于 optimizer mini-batch。

---

## 12. worker 怎样切 mini-batch 与 micro-batch

### 12.1 trainer 把元信息附到 batch

actor update 前，V1 trainer 写入：

```python
extra_info = {
    "global_batch_size": ppo_mini_batch_size,
    "mini_batch_size": ppo_mini_batch_size,
    "epochs": actor.ppo_epochs,
    "seed": actor.data_loader_seed,
    "dataloader_kwargs": {"shuffle": actor.shuffle},
    "temperature": rollout.temperature,
}
```

见
[`trainer_base.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L1692-L1705)。

### 12.2 TrainingWorker 按 epoch 产生 mini-batch

[`TrainingWorker.train_mini_batch`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/engine_workers.py#L241-L333) 会：

1. 把 global mini-batch size 除以 DP size，得到每个 rank 的 local mini-batch size。
2. 构造带 `epochs` 和 `shuffle` 的 iterator。
3. 对每个 mini-batch 调一次 `train_batch()`。
4. 每个 `train_batch()` 最终做一次 optimizer step。

关键代码：

```python
mini_batch_size_per_gpu = mini_batch_size // data_parallel_size
dataloader = make_iterator(
    data,
    mini_batch_size=mini_batch_size_per_gpu,
    epochs=epochs,
    ...,
)

for mini_batch in dataloader:
    train_batch(mini_batch)
```

### 12.3 engine 再切 micro-batch

训练 worker 把：

```text
use_dynamic_bsz
max_token_len_per_gpu
micro_batch_size_per_gpu
```

注入 engine，见
[`engine_workers.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/engine_workers.py#L335-L362)。engine 才是实际切 micro-batch、forward/backward 和 gradient accumulation 的层。

---

## 13. 为什么 micro-batch 改了，loss 理论上仍应相同

如果每个 micro-batch 都自己做 `loss.mean()`，把一个 mini-batch 切成不同数量的小块会改变梯度权重。verl 为此在 engine 中先计算完整 global mini-batch 的有效 token 数：

```python
batch_num_tokens = data["loss_mask"].sum()
all_reduce(batch_num_tokens, group=data_parallel_group)
```

FSDP 实现见
[`transformer_impl.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/engine/fsdp/transformer_impl.py#L700-L715)。随后 `ppo_loss` 把以下全局信息交给 `agg_loss`：

```text
dp_size
batch_num_tokens
global_batch_size
loss_scale_factor
```

见
[`workers/utils/losses.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/utils/losses.py#L64-L81)。每个 micro-batch 贡献的是完整 global mini-batch objective 的一部分，gradient accumulation 后再合成完整梯度。

实际浮点加法顺序、dropout/RNG、dynamic packing 仍可能带来极小数值差异，但 micro-batch 不应被当作改变算法 batch size 的手段。

---

## 14. 两套推荐的最小配置心智模型

下面只展示理解算法所需字段，不是可直接启动的完整训练命令。

### 14.1 经典 PPO + GAE

```yaml
algorithm:
  adv_estimator: gae
  gamma: 1.0
  lam: 0.95
  use_kl_in_reward: true
  kl_penalty: kl
  kl_ctrl:
    type: fixed
    kl_coef: 0.001

actor_rollout_ref:
  rollout:
    n: 1
  actor:
    policy_loss:
      loss_mode: vanilla
    clip_ratio: 0.2
    ppo_mini_batch_size: 64
    ppo_micro_batch_size_per_gpu: 2
    ppo_epochs: 1
    use_kl_loss: false

critic:
  enable: true
  ppo_mini_batch_size: 64
  ppo_micro_batch_size_per_gpu: 2
  ppo_epochs: 1
```

数据流：

```text
terminal score
 -> reward-side reference KL
 -> token rewards
 -> critic values + GAE
 -> clipped value update
 -> clipped policy update
```

### 14.2 Tool Agent GRPO + direct KL loss

```yaml
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: true
  use_kl_in_reward: false

actor_rollout_ref:
  rollout:
    n: 8
    agent:
      default_agent_loop: tool_agent
    multi_turn:
      enable: true
  actor:
    policy_loss:
      loss_mode: vanilla
    clip_ratio: 0.2
    loss_agg_mode: token-mean
    ppo_mini_batch_size: 32
    use_dynamic_bsz: true
    ppo_max_token_len_per_gpu: 24576
    ppo_epochs: 1
    use_kl_loss: true
    kl_loss_coef: 0.001
    kl_loss_type: low_var_kl

critic:
  enable: false
```

数据流：

```text
8 条同 prompt trajectory 的 terminal scores
 -> group mean/std normalization
 -> 每条 trajectory 一个 scalar advantage
 -> 广播到 assistant tokens，tool observations mask 掉
 -> PPO clipped policy loss
 -> 额外加 current-vs-reference KL loss
```

这个配置同时说明了两个容易误解的点：

1. GRPO 可以使用 PPO clipped policy loss。
2. `ppo_mini_batch_size=32` 在当前 trainer 中会乘 `rollout.n=8`，worker 实际 mini-batch 是 256 条标准 trajectory。

---

## 15. 训练指标怎样读

vanilla policy loss 返回的核心指标在
[`core_algos.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L1365-L1374)。

| 指标 | 含义 | 常见异常信号 |
|---|---|---|
| `actor/pg_loss` | 聚合后的 policy-gradient loss | 只看正负没有意义，要结合 advantage scale |
| `actor/pg_clipfrac` | 使用 clipped surrogate 的 token 比例 | 长期很高：更新过大、epoch/学习率可能太激进；长期 0：policy 几乎没移动 |
| `actor/ppo_kl` | current 与 PPO old policy 的采样近似差异 | 不是 reference KL；突增说明 PPO update 远离 anchor |
| `actor/pg_clipfrac_lower` | dual-clip 对负 advantage 生效比例 | 异常高可能有极端 ratio 或大量负 advantage |
| `actor/entropy_loss` | mask 内平均 entropy | 快速塌缩可能意味着探索消失 |
| `actor/kl_loss` / `actor/kl_coef` | loss-side reference KL 与系数 | KL 远大于 PG loss 时可能主导训练 |
| `actor/reward_kl_penalty` | reward-side 当前 KL 估计 | 与 `actor/ppo_kl` 对比对象不同 |
| `actor/reward_kl_penalty_coeff` | reward-side 当前 $\beta$ | adaptive controller 下会变化 |
| `critic/vf_loss` | value regression loss | 长期很大说明 critic/return scale 不匹配 |
| `critic/vf_clipfrac` | value clipping 生效比例 | 很高说明 critic 更新步幅过大 |

不要用单一 loss 曲线判断 RL 是否成功。至少同时看：

```text
reward distribution
advantage distribution
response length
ratio / clip fraction
old-current PPO KL
reference KL
entropy
validation outcome
```

---

## 16. 最容易犯的九个错误

1. **把 `adv_estimator=grpo` 当成 loss 也自动变成 GRPO。** 事实上默认仍 dispatch 到 `vanilla` PPO loss。
2. **把 `actor/ppo_kl` 当成 reference KL。** 它比较 current 与 PPO old；reference KL 是另一条路径。
3. **同时开 reward KL 与 loss KL，却以为只加了一次约束。** 两者发生在不同阶段，会叠加。
4. **用 micro-batch 代表有效训练 batch。** optimizer step 的单位是 mini-batch；micro-batch 只是梯度累积切片。
5. **忘记当前 trainer 会把配置的 PPO mini-batch 乘 `rollout.n`。** 算显存和 optimizer step 数时要用实际 trajectory 数。
6. **认为 `ppo_epochs` 只是重复 forward。** 每个 mini-batch 都会 optimizer step；同一批数据会被更新后的 policy 重算 log-prob。
7. **忽略 `loss_agg_mode` 的长度权重。** `token-mean` 与 `seq-mean-token-mean` 优化的加权目标不同。
8. **把 tool observation 当作模型动作。** 它参与 attention，但 `response_mask=0`，不进入 actor loss。
9. **只调 clip ratio，不检查 advantage。** 如果 group 全同分或 reward 被 KL 淹没，PPO clip 再合理也没有可学信号。

---

## 17. 建议的源码阅读顺序

按这个顺序阅读，公式与执行层比较容易对上：

1. [`trainer/ppo/v1/trainer_base.py::_step_once`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L536-L586)：先看一次 update 的全景。
2. [`trainer/ppo/v1/trainer_base.py::_compute_old_log_prob`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L1479-L1538)：理解 $\pi_{old}$。
3. [`workers/utils/losses.py::ppo_loss`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/utils/losses.py#L57-L144)：看 actor 总 loss 怎样组装。
4. [`trainer/ppo/core_algos.py::compute_policy_loss_vanilla`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L1283-L1374)：逐行对 PPO ratio 与 clipping。
5. [`trainer/ppo/core_algos.py::agg_loss`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L1138-L1204)：理解长度与全局归一化。
6. [`workers/utils/losses.py::value_loss`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/utils/losses.py#L147-L205)：理解 critic target 与 value clipping。
7. [`trainer/ppo/v1/trainer_base.py::_update_actor`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L1672-L1705)：看 trainer 传入 mini-batch/epoch 元信息。
8. [`workers/engine_workers.py::train_mini_batch`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/engine_workers.py#L241-L333)：看 epoch 与 mini-batch iterator。
9. [`workers/engine/base.py::train_batch`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/engine/base.py#L113-L130)：确认一次 mini-batch 只有一次 optimizer step。
10. [`workers/engine/utils.py::prepare_micro_batches`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/engine/utils.py#L83-L121)：最后看固定/动态 micro-batch。

读完这两章后，你应该能沿着一条具体 trajectory 回答：它的 score 放在哪里、怎样变成 advantage、哪些 token 有梯度、PPO ratio 比较哪两个 policy、一次 trainer step 到底执行了多少次 optimizer update。
