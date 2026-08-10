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

**这四个符号怎样读：** 希腊字母 $\pi$（读作“派”）表示一个 policy，也就是“看到当前上下文后，给每个候选 token 分配多少概率”的规则。下标不是乘法：`rollout`、`old`、`ref` 分别标记生成策略、PPO 的旧策略和冻结参考策略；$\theta$（读作“西塔”）是 actor 当前所有可训练参数组成的向量，因此 $\pi_\theta$ 表示“由当前参数决定的策略”。后文再次出现这四个写法时，都沿用这里的含义。

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

即只有 $\pi_{rollout}$ 与 $\pi_\theta$ 两个动态 policy：前者仍是上表中负责生成 token 的策略，后者仍是参数会被 optimizer 更新的当前策略。对应逻辑见
[`trainer_base.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L1480-L1493)。bypass mode 还可以在 PPO-clip 和显式 REINFORCE+importance-sampling loss 之间选择；这是高级 off-policy 主题，不是普通 PPO 必需配置。

---

## 3. 从 log-probability 到 PPO ratio

对第 $t$ 个已采样 token $a_t$：

$$
\log p_{old,t}=\log\pi_{old}(a_t\mid s_t),
\qquad
\log p_{\theta,t}=\log\pi_\theta(a_t\mid s_t).
$$

**公式含义：** 这两式都在问：“给定生成到第 $t$ 步时已有的上下文，某个 policy 给实际采样出的 token 多大概率？”左式记录旧策略的答案，右式记录当前可训练策略的答案；代码为了便于相减和避免极小概率下溢，保存的是概率的对数。

**符号说明：**

- $t$ 是 response 中的 token 位置；下标 `old,t` 或 `θ,t` 同时说明“来自哪个策略、对应哪个位置”。
- $a_t$ 是第 $t$ 步已经采样出来的动作；在语言模型中，动作就是选中的 token。
- $s_t$ 是选择 $a_t$ 前模型已经看到的状态，即 prompt 加上此前 token；竖线 $\mid$ 读作“在……条件下”。
- $\pi_{old}(a_t\mid s_t)$ 和 $\pi_\theta(a_t\mid s_t)$ 是两个 policy 分别给该 token 的条件概率，取值在 0 到 1 之间。
- $p_{old,t}$、$p_{\theta,t}$ 是上述两个概率的简写；$\log$ 是自然对数。$\theta$ 仍表示当前 actor 的参数。
- `\qquad` 只负责在排版中增加间距，不参与计算；逗号表示两条定义并列成立。

PPO 的 importance ratio 是：

$$
\rho_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}
=\exp(\log p_{\theta,t}-\log p_{old,t}).
$$

**公式含义：** $\rho_t(\theta)$（读作“rho，下标 t”）比较当前策略与旧策略对同一个已采样 token 的概率。分数大于 1 表示当前策略更愿意选它，小于 1 表示当前策略更不愿意选它，等于 1 表示概率没变。最后一个等号使用恒等式“两个概率之比 = 两个对数之差再取指数”。

**符号说明：**

- $\rho$ 是希腊字母 rho；下标 $t$ 指第 $t$ 个 token，括号中的 $\theta$ 强调该比值会随当前 actor 参数变化。
- 分数线表示当前概率除以旧概率；分子、分母中的 $a_t$、$s_t$、$\pi_\theta$、$\pi_{old}$ 沿用上一组公式的定义。
- $\exp(x)$ 是指数函数 $e^x$；这里的输入 $x$ 是两项 log-probability 的差。
- 减号表示用当前 log-prob 减去旧 log-prob；因为 $\log x-\log y=\log(x/y)$，再取 $\exp$ 就还原成概率比。

源码对应：

```python
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
ratio = torch.exp(negative_approx_kl)
```

见
[`compute_policy_loss_vanilla`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L1283-L1374)。上面的概率比是数学定义；实现先把原始 log-ratio 裁剪到 `[-20, 20]`，再计算 `ratio`。因此只有原始 log-ratio 落在该区间内时，代码值才与未裁剪公式完全相同。这个 clamp 不仅防止 `exp` 溢出，也把有效 ratio 限制在 `exp(-20)` 到 `exp(20)`；落在饱和区外时，梯度不会继续穿过 clamp 回传到原始 log-ratio。

还要区分“PPO ratio”和“采样分布校正”：trajectory 中的动作实际来自 $\pi_{\mathrm{rollout}}$，而 $\rho_t=\pi_\theta/\pi_{\mathrm{old}}$ 只负责约束当前 actor 相对 proximal anchor 的更新，并不会自动校正 $\pi_{\mathrm{rollout}}$ 与 $\pi_{\mathrm{old}}$ 的差异。只有显式启用 `rollout_is` 时，verl 才会在 clipped policy-gradient 项之后再乘近似的 $\pi_{\mathrm{old}}/\pi_{\mathrm{rollout}}$ 权重；这个额外权重不乘到 `ppo_kl`、entropy 或 reference-KL 项上。这里三条分数线都表示“前一个 policy 给已采样 token 的概率除以后一个 policy 的概率”，三个 policy 的角色沿用第 2 节。

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

**公式含义与符号说明：** 这里把上面的 ratio 公式代入一组具体数字。$e$ 是自然常数（约 2.718），上标表示指数；$-(-1.2)$ 等于加 1.2，所以指数为 0.2；$\approx$ 表示“约等于”，因为 1.221 是四舍五入后的近似值。这里省略下标的 $\rho$ 仍指当前概率除以旧概率。

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

**公式含义：** 这条式子为第 $t$ 个 token 计算 PPO 的“保守收益”。它在未裁剪收益与裁剪后收益中取较小者，使策略不能仅靠把概率比推得很远来持续增大目标；训练时希望把这个 $L^{clip}_t$ 最大化。

**符号说明：**

- $L$ 表示 objective（目标值）；上标 `clip` 是名称标签，不是幂；下标 $t$ 表示第 $t$ 个 token；括号中的 $\theta$ 表示它依赖当前 actor 参数。
- $\rho_t$ 是上一节的当前概率/旧概率之比；$A_t$ 是第 $t$ 个 token 的 advantage。$A_t>0$ 表示这次动作比基准好，$A_t<0$ 表示比基准差。
- 两项中的相乘表示用 advantage 决定“提高还是降低该 token 概率”以及力度。
- $\min(x,y)$ 返回 $x$、$y$ 中较小的一个。
- $\operatorname{clip}(x,l,u)$ 把 $x$ 限制在下界 $l$ 与上界 $u$ 之间：低于下界就取下界，高于上界就取上界。
- $\epsilon$（epsilon）是允许 ratio 偏离 1 的裁剪幅度，通常是小正数；$1-\epsilon$ 与 $1+\epsilon$ 分别是下、上界。
- `\left(`、`\right)` 只是让 KaTeX 自动调整括号大小，不是额外运算。

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

advantage estimator 给出的信号认为这个动作的回报低于相应 baseline 或 group reference，而当前 policy 反而相对旧 policy 提高了它的概率；PPO 会继续强烈惩罚。旧 policy 在这里提供的是 ratio 分母，并不负责判断动作好坏。clip 不是“ratio 超出区间就一律截断”，它只截断会让 surrogate objective 继续变好的方向。

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

- `clip_ratio_low` 控制下界 $1-\epsilon_{low}$；$\epsilon_{low}$ 是允许 ratio 向 1 以下移动的幅度，下标 `low` 只是“下侧”的标签。
- `clip_ratio_high` 控制上界 $1+\epsilon_{high}$；$\epsilon_{high}$ 是允许 ratio 向 1 以上移动的幅度，下标 `high` 只是“上侧”的标签。
- `clip_ratio_c` 对应 $C_{\text{clip}}>1$，会对负 advantage 再加入 dual-clip 下界；$C_{\text{clip}}$ 是额外阈值，下标 `clip` 表示它属于 dual-clip，$>$ 表示“严格大于”。

当前 dual-clip 最大化的形式可写成：

$$
L_t^{dual}=
\begin{cases}
L_t^{clip}, & A_t\ge0,\\
\max(L_t^{clip}, C_{\text{clip}} A_t), & A_t<0.
\end{cases}
$$

**公式含义：** 这是一个分情况规则：advantage 非负时仍使用普通 PPO clip；advantage 为负时，再把普通 clip 目标与 $C_{\text{clip}}A_t$ 比较并取较大者，以限制极端负目标。两行只会命中一行，不是把两行相加。

**符号说明：**

- $L_t^{dual}$ 是第 $t$ 个 token 的 dual-clip 目标；上标 `dual` 是算法名称标签。$L_t^{clip}$ 是 4.1 节已经解释的普通裁剪目标。
- $\begin{cases}\cdots\end{cases}$ 表示分段函数；每行逗号右侧是采用该行的条件。
- $A_t\ge0$ 表示 advantage 大于或等于 0，$A_t<0$ 表示 advantage 小于 0。
- $\max(x,y)$ 返回两者中较大的一个；$C_{\text{clip}}A_t$ 表示 dual-clip 阈值与 advantage 相乘，$C_{\text{clip}}>1$ 的定义沿用上方配置说明。第 10.5 节还会出现另一个归一化常数 $C_{\text{norm}}$，两者没有关系。
- `&` 和 `\\` 是 LaTeX 的对齐、换行标记，不参与数学计算。

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
+c_{\mathrm{KL}}\,\widehat{\mathcal K}_{\mathrm{ref}}.
$$

**公式含义：** actor 最终最小化的 loss 由三部分相加：policy-gradient 主损失、带负号的 entropy 奖励、带正号的 sampled reference-policy penalty。负号让更高 entropy 降低总 loss，正号让采样动作在当前 policy 与 reference policy 下的差异提高总 loss。把最后一项写成通用的 $\widehat{\mathcal K}_{\mathrm{ref}}$，是因为它是否能严格解释成某一方向的 KL，取决于 estimator 类型与动作的实际采样分布。

**符号说明：**

- $\mathcal L$ 表示要最小化的 loss；花体只是书写习惯。下标 `actor` 表示总 actor loss，下标 `PG` 表示 policy-gradient 部分。
- $c_H$ 是 entropy coefficient，控制探索奖励强度；$H(\pi_\theta)$ 是当前策略经 rollout temperature 缩放 logits 后得到的分布之 entropy。二者相乘，前面的减号表示从 loss 中扣除。
- $c_{\mathrm{KL}}$ 是 reference penalty coefficient，控制 reference 约束强度；直立下标 `KL` 是名称标签，不表示 $K$ 与 $L$ 相乘。
- $\widehat{\mathcal K}_{\mathrm{ref}}$ 是在采样 token 上计算并聚合的 reference-policy penalty；帽子表示它是采样量，花体 $\mathcal K$ 是本章为避免过度声称“严格 KL”而采用的通用记号，下标 `ref` 表示比较基准是冻结 reference policy。
- 公式中的 `\,` 只是很小的排版空格，不参与乘法；相邻的系数与函数默认表示相乘。

V1 中状态和动作来自 $\pi_{\mathrm{rollout}}$ 生成的 trajectory。默认没有把 rollout importance-sampling 权重应用到 entropy 或 reference penalty，因此最后两项应理解为“在当前采样数据上计算的正则项”；第 7.3 节会说明何时其期望才等于严格的 KL divergence。

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

如果真要用无 clipping 的 $-A\log\pi$ loss（$A$ 是 advantage，$\pi$ 是策略给已采样动作的概率，$\log$ 是自然对数，前置负号把“最大化收益”改写为优化器可最小化的 loss），当前专门实现是
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

**公式含义：** 在第 $t$ 个生成位置，把当前策略对词表中每个候选 token 的“概率 × 对数概率”加起来并取负，得到该位置的不确定程度。概率分布越均匀，entropy 越大；几乎只给一个 token 概率时，entropy 越小。verl 计算这里的 $\pi_\theta$ 时，会先用 rollout temperature 缩放 actor logits，因此它是训练 forward 中实际使用的 temperature-scaled 分布。

**符号说明：**

- $H_t$ 是第 $t$ 个 token 位置的 entropy；括号中的 $\pi_\theta$ 说明 entropy 来自当前 actor policy。
- $\sum_a$ 表示对所有可能动作 $a$ 求和；这里一个动作就是词表中的一个候选 token，而不是只对实际采样的 token 求和。
- $\pi_\theta(a\mid s_t)$ 是当前 actor 的 logits 经 rollout temperature 缩放后，在状态 $s_t$ 下选择候选 $a$ 的概率；$\mid$ 表示条件关系。
- $\log\pi_\theta(a\mid s_t)$ 是该概率的自然对数；概率小于 1 时其对数非正，所以最前面的负号使 entropy 非负。
- 下标 $t$ 仍是 response 位置，$\theta$ 仍是当前可训练参数。乘积没有写 $\times$，而是按数学惯例把两项并排书写。

entropy 高，说明概率分布更平；entropy 低，说明模型几乎确定只选少数 token。

verl 最小化：

$$
\mathcal L_{actor}=\mathcal L_{PG}-c_H H.
$$

**公式含义与符号说明：** 这是上一节 actor 总 loss 去掉 reference penalty 后的简写。$\mathcal L_{actor}$ 是待最小化的 actor loss，$\mathcal L_{PG}$ 是 policy-gradient loss，$H$ 是按当前 `loss_agg_mode` 把有效 response token 的 $H_t$ 聚合后的 entropy，$c_H$ 是非负权重；减号意味着聚合后的 entropy 越大，总 loss 越小，因此 optimizer 会保留更多探索。只有 mean 类 aggregation 才能把 $H$ 简称为“mask 内平均 entropy”。

所以 `entropy_coeff > 0` 会奖励更高 entropy。配置是：

```yaml
actor_rollout_ref:
  actor:
    entropy_coeff: 0.0
    calculate_entropy: false
```

如果 `entropy_coeff != 0`，trainer 会自动请求 actor forward 计算 entropy，见
[`trainer_base.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L1672-L1703)。entropy 同样只让 `response_mask` 选中的 assistant token 参与聚合；具体是平均、求和还是序列归一化，则由 `loss_agg_mode` 决定。

---

## 7. 两条完全不同的 reference KL 路径

KL regularization 的目的，是不让 post-trained policy 偏离冻结 reference policy 太远。verl 提供两条路径。

| 对比 | reward-side KL | loss-side KL |
|---|---|---|
| 开关 | `algorithm.use_kl_in_reward` | `actor_rollout_ref.actor.use_kl_loss` |
| 使用的 policy | `old_log_probs` vs `ref_log_prob` | 当前 `log_prob` vs `ref_log_prob` |
| 发生时机 | advantage 计算前 | 每次 actor forward/backward |
| 数学形式 | $r_t=\mathrm{score}_t-\beta\widehat D_{\mathrm{KL},t}$ | $\mathcal L\mathrel{+}=c_{\mathrm{KL}}\widehat{\mathcal K}_{\mathrm{ref},t}$ |
| 会不会改变 `token_level_rewards` | 会 | 不会 |
| 系数 | fixed/adaptive KL controller | 固定 `kl_loss_coef` |
| 会不会间接改变 advantage | 会 | 不会，直接改变梯度 |

表中两条紧凑公式的符号如下：$r_t=\mathrm{score}_t-\beta\widehat D_{\mathrm{KL},t}$ 表示第 $t$ 个 token 的训练 reward 等于原始 score/reward 减去 sampled KL 罚分，其中 $r_t$ 是处理后的 reward，$\mathrm{score}_t$ 是未罚分的信号，$\beta$ 是罚分系数，$\widehat D_{\mathrm{KL},t}$ 是该 token 的策略差异估计；$\mathcal L\mathrel{+}=c_{\mathrm{KL}}\widehat{\mathcal K}_{\mathrm{ref},t}$ 表示把 $c_{\mathrm{KL}}$ 倍的 reference penalty 加到现有 loss 中，$\mathrel{+}=$ 是“在原值上累加”，$\mathcal L$ 是 loss，$\widehat{\mathcal K}_{\mathrm{ref},t}$ 沿用第 5 节的通用 sampled penalty 记号。两式的下标 $t$ 都表示 token 位置，直立下标 `KL`、`ref` 都是名称标签。使用直立的 $\mathrm{score}$，是为了不与表示状态的 $s_t$ 混淆。

### 7.1 reward-side KL

实现位于
[`apply_kl_penalty`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/ray_trainer.py#L78-L117)：

$$
r_t=\mathrm{score}_t-\beta\widehat D_{\mathrm{KL},t}.
$$

**公式含义：** reward-side 路径先从第 $t$ 个 token 的原始训练信号中扣除 reference KL 罚分，再把所得 $r_t$ 交给 advantage estimator；策略偏离 reference 越大，扣分通常越多。

**符号说明：** $r_t$ 是 KL shaping 后的 token reward；$\mathrm{score}_t$ 是 shaping 前的 token score/reward，直立英文标签用于区别第 3 节中代表状态的 $s_t$；$\beta$（beta）是非负 KL 系数；$\widehat D_{\mathrm{KL},t}$ 是第 $t$ 个 token 的 sampled KL estimator。帽子表示采样估计，直立下标 `KL` 指差异度量类型，下标 $t$ 指 token 位置；减号表示把罚分扣掉，$\beta\widehat D_{\mathrm{KL},t}$ 表示两者相乘。默认 `kl`/`k1` 时，它具体等于 `old_log_probs - ref_log_prob`；单个 token 的值可以为负，只有满足第 7.3 节的采样条件时，其期望才是严格 KL。

若 controller 为 adaptive，系数更新为：

$$
e=\operatorname{clip}\left(
\frac{\mathrm{KL}_{\mathrm{current}}}
{\mathrm{KL}_{\mathrm{target}}}-1,-0.2,0.2
\right),
$$

$$
\beta\leftarrow\beta
\left(1+e\frac{n_{\mathrm{seq}}}{h_{\mathrm{ctrl}}}\right).
$$

**两条公式的含义：** 第一式把“当前 KL 相对目标 KL 高了或低了多少”压缩到 $[-0.2,0.2]$，得到本次控制误差 $e$。第二式用这个误差缓慢放大或缩小旧的 $\beta$：当前 KL 高于目标时 $e>0$，新系数变大；低于目标时 $e<0$，新系数变小。

**符号说明：**

- $e$ 是 controller 的有符号误差，不是自然常数；$\mathrm{KL}_{\mathrm{current}}$ 是观测到的当前 KL 统计量，$\mathrm{KL}_{\mathrm{target}}$ 是期望维持的目标 KL。直立下标 `current`、`target` 是角色标签。
- 分数 $\mathrm{KL}_{\mathrm{current}}/\mathrm{KL}_{\mathrm{target}}$ 是“当前值是目标值的几倍”，减 1 后，恰好等于目标时结果为 0。实现先对每条 trajectory 的有效 token 做 masked mean，再对 trajectory 等权平均得到 $\mathrm{KL}_{\mathrm{current}}$，并非把全 batch 所有 token 直接混在一起平均。
- $\operatorname{clip}(x,-0.2,0.2)$ 把 $x$ 限制在 -0.2 到 0.2；方括号 $[-0.2,0.2]$ 表示包含两端点的区间。
- $\beta\leftarrow\beta(\cdots)$ 中的左箭头表示“用右侧结果更新左侧变量”，不是普通等号；右侧第一个 $\beta$ 是更新前的旧系数。
- $n_{\mathrm{seq}}$ 是这次更新中 batch 第一维的 trajectory/sequence 数；运行时它被传给名为 `n_steps` 的参数，但不是 optimizer step 数。$h_{\mathrm{ctrl}}$ 是配置 `horizon` 给出的调节时间尺度，直立下标 `ctrl` 表示 controller。
- 括号内的 1 是“不改变系数”的基准倍率；$e\,n_{\mathrm{seq}}/h_{\mathrm{ctrl}}$ 是在该基准上加减的相对调整量。相邻书写表示乘法。

对应源码在
[`AdaptiveKLController`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L153-L175)。KL 高于 target 时 $\beta$ 增大，低于 target 时减小。V1 当前在 rollout correction/rejection 之前计算这个统计量并更新 controller；不要把它与 correction 后留下样本的 KL 混为一谈。

### 7.2 loss-side KL

actor forward 得到当前 `log_prob` 后：

```python
kld = kl_penalty(log_prob, ref_log_prob, config.kl_loss_type)
kl_loss = agg_loss(kld, response_mask, config.loss_agg_mode, ...)
policy_loss += config.kl_loss_coef * kl_loss
```

见
[`workers/utils/losses.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/utils/losses.py#L131-L143)。因为 $\pi_\theta$ 会在每个 mini-batch optimizer step 后改变，所以 loss-side KL 会持续追踪最新 actor；这里的 $\pi$ 是 policy，参数下标 $\theta$ 表示当前可训练 actor，与第 2 节定义相同。

### 7.3 KL estimator 在代码中的形式

令：

$$
d=\log\pi(a\mid s)-\log\pi_{\mathrm{ref}}(a\mid s),
\qquad
\widetilde d=\operatorname{clip}(d,-20,20).
$$

**公式含义：** 对同一个已采样动作，先算被比较 policy 与冻结 reference policy 的 log-probability 之差，并把这个差记作 $d$；低方差 `k3` 分支还会把 $d$ 裁剪到 -20 至 20，得到 $\widetilde d$。reward-side 中前一项来自 old policy，loss-side 中前一项来自当前 actor；具体由调用路径决定。

**符号说明：** $d$ 是单个 token 的原始 log-ratio；$\pi$ 是当前调用要约束的 policy，$\pi_{\mathrm{ref}}$ 是冻结 reference policy；$a$ 是已采样动作/token，$s$ 是采样前上下文，$\mid$ 表示“在状态 $s$ 的条件下”；直立下标 `ref` 是 reference 标签。$\log$ 是自然对数，减号表示前一个 policy 的 log-prob 减去 reference log-prob。波浪号表示经过数值裁剪后的 $\widetilde d$；$\operatorname{clip}$ 的三个参数依次是输入、下界和上界。这里没有位置下标，是为了写得简洁，`\qquad` 仅增加排版间距。

[`kl_penalty_forward`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L2213-L2248) 支持：

| 配置 | token estimator |
|---|---|
| `kl` / `k1` | $d$ |
| `abs` | $\lvert d\rvert$ |
| `mse` / `k2` | $\tfrac12 d^2$ |
| `low_var_kl` / `k3` | $\operatorname{clip}(e^{-\widetilde d}+\widetilde d-1,-10,10)$ |
| `full` | 当前函数中未实现 |

**表中各公式怎样读：**

- `k1` 直接使用 $d$；$d$ 就是上式定义的 log-prob 差，没有额外运算。
- `abs` 使用 $\lvert d\rvert$；两侧竖线表示绝对值，也就是忽略正负号，只保留 $d$ 离 0 多远。
- `k2` 使用 $\tfrac12 d^2$；$\tfrac12$ 是二分之一，右上角 2 表示平方，两者相乘后会更强地惩罚较大的偏差。
- `k3` 先使用上式的 $\widetilde d$，再计算 $e^{-\widetilde d}+\widetilde d-1$ 并把结果裁剪到 -10 至 10；这里的 $e$ 是自然常数，负号位于指数中。未裁剪表达式在 $\widetilde d=0$ 时等于 0；外层 clamp 还限制极端值，并使饱和区梯度为 0。
- `k1`、`k2`、`k3` 是 estimator 名称，不是公式里的下标；表中的每条 estimator 都为一个 token 产生一个标量罚分。

`k1` 单个样本可以为负。只有当动作确实按公式第一项 policy 采样，即 $a\sim\pi$ 时，未裁剪 `k1` 的期望才等于 $D_{\mathrm{KL}}(\pi\Vert\pi_{\mathrm{ref}})$；其中 $\sim$ 表示“采样自”，$\Vert$ 表示 KL 的方向分隔，$D_{\mathrm{KL}}$ 是 Kullback–Leibler divergence。未裁剪 `k3` 也利用同一采样条件得到该方向 KL 的低方差形式，而 `abs` 与 `k2` 本身不是严格 KL estimator。V1 trajectory 通常来自 $\pi_{\mathrm{rollout}}$，它可能不同于 reward-side 的 $\pi_{\mathrm{old}}$ 或 loss-side 的 $\pi_\theta$；且 reference penalty 不乘 rollout IS 权重，所以当前样本平均不能无条件称为上述严格 KL。`k1+`、`k3+` 等带 `+` 形式会用 straight-through 技巧，让前向值与反向梯度采用不同 estimator，见
[`kl_penalty`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L2185-L2210)。

### 7.4 两条 KL 可以同时开，但通常要有意为之

只要任一开关启用，trainer 就会创建/使用 reference policy，见
[`need_reference_policy`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/utils.py#L75-L80)。两条路径同时开启不会被禁止，但配置校验会打印 notice，见
[`utils/config.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/utils/config.py#L169-L170)。

同时启用等于既改变 reward/advantage，又在 loss 上再施加一次 reference regularization。除非你明确设计了这个目标，否则很容易把 KL 约束加重两次。

### 7.5 `actor/ppo_kl` 不是 reference KL

vanilla loss 里的指标：

```python
raw_log_ratio = log_prob - old_log_prob
clamped_log_ratio = torch.clamp(raw_log_ratio, min=-20.0, max=20.0)
ppo_kl = masked_mean(-clamped_log_ratio, response_mask)
```

测的是当前 policy 与 **PPO old policy** 的裁剪后采样近似差异，用来观察一次 PPO update 移动了多远。负号表示把“current 减 old”的 clamped log-ratio 改写成“old 减 current”，`masked_mean` 只平均有效 action token；若原始 log-ratio 越过 `[-20, 20]`，指标也会饱和。它不是 $\pi_\theta$ 与 $\pi_{\mathrm{ref}}$ 的 reference penalty：$\pi_\theta$ 是参数下标为 $\theta$ 的当前策略，$\pi_{\mathrm{ref}}$ 是下标标记为 `ref` 的参考策略，两者都沿用第 2 节定义。变量都叫 KL，但回答的是两个不同问题。

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

**公式含义：** 先以旧 critic 预测为中心画出一个允许区间，再把新预测限制在这个区间内，得到裁剪后的预测。新预测若没有越界就保持不变，越界则只取最近的边界。

**符号说明：**

- $V$ 表示 value prediction，即 critic 对“从当前状态继续下去能获得多少回报”的估计；下标 $t$ 是 token 位置。
- 上标 `new`、`old`、`clip` 都是角色标签，不是乘方，分别表示当前新预测、更新前保存的旧预测和裁剪结果。
- $\epsilon_v$（epsilon-v）是 critic 允许的最大变化幅度；下标 $v$ 表示它属于 value loss。
- $V^{old}_t-\epsilon_v$ 和 $V^{old}_t+\epsilon_v$ 是允许区间的下、上界。
- $\operatorname{clip}(x,l,u)$ 把输入 $x$ 限制到下界 $l$ 与上界 $u$ 之间，含义与 policy ratio 中的 clip 相同。

再取两个平方误差中更大的一个。以下先写默认 `token-mean` aggregation 的形式：

$$
\mathcal L_V
=\frac12\,\mathbb E_t\left[
\max\left(
(V^{new}_t-G_t)^2,
(V^{clip}_t-G_t)^2
\right)
\right].
$$

**公式含义：** 在 `token-mean` 模式下，对每个有效 token 同时计算“新预测对 target 的平方误差”和“裁剪后预测对 target 的平方误差”，取更大的那个，再对 token 求平均并乘二分之一。取较大误差可防止 critic 通过一次跨出允许区间的大跳跃来获得过小 loss。

**符号说明：**

- $\mathcal L_V$ 是 critic 的 value loss；花体 $\mathcal L$ 表示 loss，下标 $V$ 表示 value 分支。
- $\frac12$ 是二分之一；它不改变最优点，只让平方项求导后的系数 2 抵消，便于书写梯度。
- $\mathbb E_t[\cdot]$ 在这条默认形式中表示对 mask 选中的 token 位置 $t$ 求平均；黑板粗体 $\mathbb E$ 表示 expectation。若 `critic.loss_agg_mode` 不是 `token-mean`，代码会用第 10 节相应的 aggregation 替换这一步。
- $G_t$ 是第 $t$ 个 token 的 return target，即上一章计算出的 `returns`；$V_t^{new}$、$V_t^{clip}$ 沿用上一式定义。
- 圆括号里的减法是 prediction error，右上角 2 是平方，所以正负误差都会产生非负惩罚。
- $\max(x,y)$ 取两个平方误差中较大的一个；方括号只是把被平均的整个表达式括起来。

实现见
[`compute_value_loss`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L2122-L2182)，worker 入口见
[`workers/utils/losses.py::value_loss`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/utils/losses.py#L147-L205)。

源码实际先生成逐 token 的 clipped error matrix，再交给通用 `agg_loss(config.loss_agg_mode, ...)`。因此上式中的 token mean 不是 critic 永远固定的算法；切换 `loss_agg_mode` 会像 actor 一样改变不同长度轨迹的权重。

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
  loss_agg_mode: ...
  loss_scale_factor: ...
```

actor 和 critic 可以有不同的 mini/micro batch 与 epoch 数。critic 的 `loss_agg_mode` 与 `loss_scale_factor` 默认继承 actor 对应设置，也可以显式覆盖。

---

## 9. `response_mask` 最终怎样进入 loss

所有关键 loss 的**分子**都遵守同一规则：

```python
masked_loss = loss_matrix * response_mask
```

aggregation 的分母还可能使用 engine 预先统计的 `batch_num_tokens = loss_mask.sum()`。通常 `loss_mask == response_mask`；但启用 rollout rejection 时，correction 会更新 `response_mask` 而保留原始 `loss_mask`，所以被拒绝 token 不进入 loss 分子，却仍可能留在 `token-mean` 的全局归一化 token 数中。下面会把这两个 mask 分开记号。

因此：

- prompt token 不在 `[B,response_length]` loss matrix 中。
- tool observation 的 `response_mask=0`，不产生 policy/value/KL/entropy loss。
- response padding 的 mask 也是 0。
- V1 为批次整除而添加的 synthetic padding sample，其整个 `response_mask` 都是 0，因此不会产生梯度，见
  [`padding_utils.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/padding_utils.py#L70-L138)。

模型输出与 response token 的对齐还需要左移一位：位置 $t$ 的 logits 预测下一个 token。这里 $t$ 是序列位置下标，不是一个需要学习的参数；“位置 $t$ 的 logits”是模型在该位置输出、用于预测位置 $t+1$ token 的未归一化分数，其中 $t+1$ 表示紧接着的下一个位置。no-padding 输出还原为 response tensor 的切片逻辑在
[`workers/utils/padding.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/utils/padding.py#L99-L143) 和
[`response_from_nested`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/utils/padding.py#L196-L212)。

---

## 10. Loss aggregation：谁在平均谁

设第 $i$ 条轨迹有效动作 token 数为 $T_i$，token loss 为 $\ell_{i,t}$，实际进入分子的 `response_mask` 为 $m_{i,t}$，engine 用于全局 token 归一化的 `loss_mask` 为 $u_{i,t}$。这里 $i$ 是 batch 中的轨迹编号，$t$ 是该轨迹中的 token 位置；$T_i$ 的下标表示“第 $i$ 条轨迹的长度”；$\ell$（小写 ell）表示尚未聚合的单 token loss；$m_{i,t}$ 与 $u_{i,t}$ 都是 0/1 开关。常规无 rejection 路径二者相同；rollout rejection 后，$m_{i,t}$ 可能被改成 0，而 $u_{i,t}$ 保留原值。逗号分隔的双下标 $(i,t)$ 表示“第 $i$ 条轨迹、第 $t$ 个位置”，不是两个数相乘。需要按轨迹数归一化时，用 $B_{\text{agg}}$ 表示 worker 实际用于 loss aggregation 的全局 trajectory mini-batch 大小；下标 `agg` 提醒它与第 11.1 节的 prompt-group 数 $B_{\text{prompt}}$ 不同。`loss_agg_mode` 会实质改变不同长度轨迹的相对权重。

实现统一在
[`agg_loss`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L1138-L1204)。

### 10.1 `token-mean`

$$
\mathcal L
=\frac{\sum_i\sum_t m_{i,t}\ell_{i,t}}
{N_{\mathrm{loss}}},
\qquad
N_{\mathrm{loss}}=\sum_i\sum_t u_{i,t}.
$$

**公式含义：** 分子把 `response_mask` 当前保留的 token loss 相加，分母使用 engine 在完整 global mini-batch 上由 `loss_mask` 统计的 token 数。通常两个 mask 相同，此式就是“每个有效 token 的平均 loss”，每个 token 等权，而 token 更多的长轨迹自然贡献更多项。启用 rollout rejection 时，两者可能不同：被拒绝 token 从分子消失，但仍保留在分母计数中，所以 rejection 还会缩小整体 loss scale。

**符号说明：** $\mathcal L$ 是聚合后的标量 loss；$\sum_i$ 表示遍历所有轨迹，$\sum_t$ 表示遍历每条轨迹的所有 token；两层求和合起来覆盖整个 batch。$m_{i,t}\ell_{i,t}$ 是 response mask 与 token loss 相乘，mask 为 0 时该项归零。$N_{\mathrm{loss}}$ 是全局 normalization token count，直立下标 `loss` 是名称标签；它等于所有 $u_{i,t}$ 的总和。$u_{i,t}$ 是 `loss_mask`，其定义与和 $m_{i,t}$ 的区别见本节开头。分数线表示分子除以分母，`\qquad` 只增加两条等式之间的排版间距。

每个有效 token 权重相等。长回答包含更多 token，因此整条轨迹的总影响通常更大。当前 actor 默认是这个模式。

### 10.2 `token-sum`

$$
\mathcal L=\sum_i\sum_t m_{i,t}\ell_{i,t}.
$$

**公式含义与符号说明：** 这次只保留上一式的分子：$\mathcal L$ 等于所有轨迹 $i$、所有位置 $t$ 上的 masked token loss 之和。两个 $\sum$ 是两层求和，$m_{i,t}$ 是 0/1 mask，$\ell_{i,t}$ 是对应 token loss；因为没有除号或归一化因子，有效 token 越多，loss 与梯度的总尺度通常越大。

不做平均，梯度大小会随有效 token 总数增长；通常只有在外部已经精确控制 scale 时使用。

### 10.3 `seq-mean-token-sum`

$$
\mathcal L
=\frac1{B_{\text{agg}}}\sum_i\sum_t m_{i,t}\ell_{i,t}.
$$

**公式含义：** 先对每条轨迹内的有效 token loss 求和，再把所有轨迹的序列和相加，最后除以 loss aggregation 使用的全局 trajectory mini-batch 大小。它平均的是“轨迹总 loss”，不是单个 token，所以长轨迹仍可能贡献更大的序列和。

**符号说明：** $B_{\text{agg}}$ 是这个优化 batch 的全局 trajectory 归一化大小，当前生产路径由实际 trajectory mini-batch 配置提供，并可能包含为整除而添加的 synthetic padding 行；这些全零 mask 行不产生梯度，但仍属于该归一化大小。$1/B_{\text{agg}}$ 是外层权重；$i$ 遍历轨迹，$t$ 遍历 token；$m_{i,t}\ell_{i,t}$、两层 $\sum$ 与 $\mathcal L$ 沿用前两式定义。

先对每条轨迹 token 求和，再对轨迹平均。长轨迹仍有更大总权重。

### 10.4 `seq-mean-token-mean`

$$
T_i=\sum_t m_{i,t},\qquad
q_i=\mathbf 1[T_i>0],
\qquad
\mathcal L
=\frac1{B_{\text{agg}}}\sum_i q_i
\frac{\sum_t m_{i,t}\ell_{i,t}}
{T_i+\varepsilon_{\text{num}}}.
$$

**公式含义：** 先数出每条轨迹的有效 token 数，并标记它是否非空；内层为非空轨迹计算“有效 token 的平均 loss”，外层再按 $B_{\text{agg}}$ 归一化。这样一条 2-token 轨迹与一条 20-token 轨迹各贡献一个序列平均值，而 synthetic 全零 mask 行会被排除，不会产生除零问题。

**符号说明：** $T_i=\sum_t m_{i,t}$ 是第 $i$ 条轨迹的有效 token 数；$q_i=\mathbf 1[T_i>0]$ 是指示函数，条件成立时取 1，否则取 0，因此全零行被屏蔽。$\varepsilon_{\text{num}}$ 是实现使用的数值稳定小量（当前为 `1e-8`），防止内层分母为 0；下标 `num` 表示 numerical stability，它不是 PPO 的裁剪宽度。内层分子是 masked token loss 总和；外层 $\sum_i$ 遍历轨迹，$1/B_{\text{agg}}$ 使用生产路径传入的全局 trajectory mini-batch 大小。$\mathcal L$、$i$、$t$、$m_{i,t}$、$\ell_{i,t}$ 都沿用本节开头定义。

每条轨迹先做 token mean，所以每条非空轨迹大致等权。这更接近原始 GRPO 的 sample-level loss，但在长 CoT 中可能改变优化稳定性。

### 10.5 `seq-mean-token-sum-norm`

$$
\mathcal L
=\frac1{B_{\text{agg}}}\sum_i
\frac{\sum_t m_{i,t}\ell_{i,t}}{C_{\text{norm}}}.
$$

**公式含义：** 每条轨迹仍先把有效 token loss 相加，但不再除以它自己的有效长度，而统一除以常数 $C_{\text{norm}}$；之后再按 $B_{\text{agg}}$ 归一化。固定 $C_{\text{norm}}$ 可以让不同 batch 的归一化尺度更一致，同时长轨迹仍因包含更多有效项而有更大总贡献。

**符号说明：** $C_{\text{norm}}$ 是所有轨迹共用的正归一化常数，不随下标 $i$ 或 $t$ 改变；下标 `norm` 说明它用于 normalization，与第 4.3 节的 dual-clip 阈值 $C_{\text{clip}}$ 无关。$1/B_{\text{agg}}$、两个求和、mask $m_{i,t}$、token loss $\ell_{i,t}$ 和总 loss $\mathcal L$ 均沿用上文。

$C_{\text{norm}}$ 来自 `actor.loss_scale_factor`；它就是上式分母中的归一化常数。若为 `null`，当前实现回退到**当前 micro-batch** 的 padded horizon。dynamic/ragged 分包产生不同宽度时，这个 horizon 可能变化，所以此模式下只有显式设置固定 `loss_scale_factor`，才能保证归一化因子不随 micro-batch 切法改变。把 $C_{\text{norm}}$ 设成固定 `max_response_length` 是 Dr.GRPO 常见设置，可避免这种漂移。现有说明见
[`docs/algo/grpo.md`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/docs/algo/grpo.md#L51-L62)。

### 10.6 一个长度例子

两条轨迹 token loss 都为 1：

```text
trajectory A: 2 个有效 token -> [1, 1]
trajectory B: 6 个有效 token -> [1, 1, 1, 1, 1, 1]
```

- `token-mean`：A 占总梯度权重的 $2/8$，B 占 $6/8$；两个分数的分母 8 是全 batch 的有效 token 总数，分子 2、6 分别是 A、B 的有效 token 数，所以它们分别等于 1/4 和 3/4。
- `seq-mean-token-sum`：A 的序列 loss 为 2，B 为 6，再求平均；仍是 1:3。
- `seq-mean-token-mean`：A、B 的序列 loss 都为 1；两条轨迹等权。

所以调 `loss_agg_mode` 不是纯性能优化，而是改变训练目标的权重结构。

---

## 11. Batch、mini-batch、micro-batch 到底是什么

这三个词最容易造成配置错误。

### 11.1 rollout/train batch

`data.train_batch_size` 记为 $B_{\text{prompt}}$，表示每个 trainer step 的原始 prompt group 数。若每个 prompt 采样：

```yaml
actor_rollout_ref:
  rollout:
    n: n
```

标准“每个 session 一个 output”的路径会产生：

$$
N_{trajectory}=B_{\text{prompt}}\times n.
$$

**公式含义：** rollout 产生的轨迹总数，等于 prompt group 数乘以每个 prompt 的采样次数。例如 128 个 prompt 各采 8 次，就得到 1024 条轨迹。

**符号说明：** $N_{trajectory}$ 是 trajectory 数量，$N$ 表示 count，下标 `trajectory` 说明数的是什么；$B_{\text{prompt}}$ 是 `train_batch_size` 给出的 prompt group 数；$n$ 是 `rollout.n`，即每个 prompt 生成多少条结果；$\times$ 是乘号。下标 `prompt` 特意把它与第 10 节的 $B_{\text{agg}}$ 区分开，三者都是计数，因此应为非负整数。

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

因此在当前标准路径中，配置 `M=32, n=8` 对应 worker 中每个 actor mini-batch 为 256 条 trajectory。可以把配置值理解成 **prompt-group 等价大小**；源码中的实际 trajectory mini-batch 是 $M\times n$，其中 $M$ 是配置的 prompt-group 等价 mini-batch 大小，$n$ 是每个 prompt 的 rollout 数，$\times$ 表示相乘，结果是 worker 实际收到的轨迹条数。

### 11.3 micro-batch

mini-batch 可能无法一次放进 GPU，于是每个 DP rank 再把本地 mini-batch 切成多个 micro-batch：

```yaml
actor_rollout_ref:
  actor:
    ppo_micro_batch_size_per_gpu: b_mu
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
- micro-batch 主要是显存/吞吐旋钮；在正确且固定的归一化下，不应改变有效 batch objective。例外是 `seq-mean-token-sum-norm` 配合 `loss_scale_factor=null`：此时分母会回退到当前 micro-batch 的 padded horizon，改变分包可能改变 objective scale。

### 11.4 一个从头算到尾的例子

配置：

```text
train_batch_size              B_prompt = 128 prompts
rollout.n                            n = 8
actor.ppo_mini_batch_size            M = 32 prompt-equivalents
data-parallel size                   D = 8 DP replicas
actor.ppo_micro_batch_size_per_gpu   b_mu = 2 trajectories/GPU
actor.ppo_epochs                         2
```

这个例子假设纯 FSDP：没有额外 TP、PP 或 SP，因此一个 DP replica 恰好对应一张 GPU。一般情况下，$D$ 表示 data-parallel ranks/replicas 的数量，不等于集群总 GPU 数；下标或字母 $D$ 在这里只是 data parallel size 的简写。

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

在每个 mini-batch 内，`old_log_probs` 保持不变；但每次 optimizer step 后 $\pi_\theta$ 都改变，所以后续 mini-batch/epoch 重新 forward 得到的 current `log_prob` 会变化。这里 $\pi$ 表示 policy，下标 $\theta$ 表示当前可训练参数；optimizer step 改变 $\theta$，也就改变了 $\pi_\theta$ 给 token 的概率。

### 11.5 dynamic batch size

若：

```yaml
actor_rollout_ref:
  actor:
    use_dynamic_bsz: true
    ppo_max_token_len_per_gpu: 24576
```

micro-batch 不再固定包含 $b_\mu$ 条 trajectory。实现先用 `ppo_max_token_len_per_gpu` 估算需要切成多少个 micro-batch，再按 attention workload（近似为每条序列长度平方之和）平衡这些分块；因此该配置是规划 budget，不保证每个最终 micro-batch 的 token 总数都严格不超过它。这里 $b_\mu$（读作“b-mu”）是上方 `ppo_micro_batch_size_per_gpu` 配置的固定轨迹条数，下标 $\mu$ 用来提示 micro-batch；它与第 10 节的 token mask $m_{i,t}$ 没有关系。启用 dynamic batching 后不再把它当作每个 micro-batch 必须恰好包含的数量。实现入口是
[`prepare_micro_batches`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/engine/utils.py#L83-L121)：

- `use_dynamic_bsz=false`：按 `micro_batch_size_per_gpu` 等数量切分。
- `use_dynamic_bsz=true`：用 `max_token_len_per_gpu` 与 sequence-parallel size 决定分块数，再按 attention workload 近似平衡。

dynamic batching 改变的是 micro-batch 形状和数量；在支持全局归一化的 aggregation 下，不改变上层 mini-batch 应代表的全局 objective。`seq-mean-token-sum-norm` 且 `loss_scale_factor=null` 仍是例外，因为其回退 horizon 可能随分块宽度变化。

### 11.6 log-prob/value inference 有独立 micro-batch

不要把训练 micro-batch 与这些 forward-only 配置混在一起：

- `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`：重算 actor old log-prob。
- `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`：计算 reference log-prob。
- actor/ref 相应的 `*_max_token_len_per_gpu`：dynamic forward token budget。
- `critic.ppo_infer_max_token_len_per_gpu`：当前 V1 接到 critic engine 的 inference token budget；固定版本的默认 YAML 未声明它，Hydra CLI 新增时需要 `+critic.ppo_infer_max_token_len_per_gpu=...`。

它们主要影响推理阶段显存和吞吐，不等于 optimizer mini-batch。`critic.forward_micro_batch_size_per_gpu` 属于 legacy 路径，当前固定版本的 V1 setup 没有把它接成有效的 fixed critic-inference micro-batch 旋钮。

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

如果每个 micro-batch 都自己做 `loss.mean()`，把一个 mini-batch 切成不同数量的小块会改变梯度权重。verl 为此在 engine 中先用 `loss_mask` 计算完整 global mini-batch 的 normalization token 数：

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
[`workers/utils/losses.py`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/utils/losses.py#L64-L81)。每个 micro-batch 贡献的是完整 global mini-batch objective 的一部分，gradient accumulation 后再合成完整梯度。这里的 `batch_num_tokens` 并不无条件等于当前 `response_mask.sum()`：rollout rejection 只改后者，所以 rejection 路径的 `token-mean` 会用较小的分子除以原始 normalization token count；这会改变 scale，但不会让同一批数据因 micro-batch 分包方式不同而失去一致性。

实际浮点加法顺序、dropout/RNG、dynamic packing 仍可能带来极小数值差异，但 micro-batch 不应被当作改变算法 batch size 的手段。唯一需要单独记住的归一化例外仍是 `seq-mean-token-sum-norm` 加 `loss_scale_factor=null`：分母使用当前 micro-batch 的 padded horizon，分包宽度不同就可能产生实质 scale 差异；若要求 invariance，应设置固定 scale factor。

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
| `actor/entropy_loss` | 按 `loss_agg_mode` 聚合的 masked entropy | 快速塌缩可能意味着探索消失；只有 mean 模式才能直接读作 mask 内平均 |
| `actor/kl_loss` / `actor/kl_coef` | loss-side reference KL 与系数 | KL 远大于 PG loss 时可能主导训练 |
| `actor/reward_kl_penalty` | reward-side 当前 KL 估计 | 与 `actor/ppo_kl` 对比对象不同 |
| `actor/reward_kl_penalty_coeff` | reward-side 当前 $\beta$；$\beta$（beta）是第 7.1 节从 reward 中扣除 KL 罚分时使用的系数 | adaptive controller 下会变化 |
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
2. [`trainer/ppo/v1/trainer_base.py::_compute_old_log_prob`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L1479-L1538)：理解 $\pi_{old}$，也就是下标 `old` 标记的 PPO 固定旧策略（$\pi$ 表示 policy）。
3. [`workers/utils/losses.py::ppo_loss`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/utils/losses.py#L57-L144)：看 actor 总 loss 怎样组装。
4. [`trainer/ppo/core_algos.py::compute_policy_loss_vanilla`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L1283-L1374)：逐行对 PPO ratio 与 clipping。
5. [`trainer/ppo/core_algos.py::agg_loss`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/core_algos.py#L1138-L1204)：理解长度与全局归一化。
6. [`workers/utils/losses.py::value_loss`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/utils/losses.py#L147-L205)：理解 critic target 与 value clipping。
7. [`trainer/ppo/v1/trainer_base.py::_update_actor`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/trainer/ppo/v1/trainer_base.py#L1672-L1705)：看 trainer 传入 mini-batch/epoch 元信息。
8. [`workers/engine_workers.py::train_mini_batch`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/engine_workers.py#L241-L333)：看 epoch 与 mini-batch iterator。
9. [`workers/engine/base.py::train_batch`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/engine/base.py#L113-L130)：确认一次 mini-batch 只有一次 optimizer step。
10. [`workers/engine/utils.py::prepare_micro_batches`](https://github.com/verl-project/verl/blob/d33ddd7140f44d392e0e10b48a8902651a1340f4/verl/workers/engine/utils.py#L83-L121)：最后看固定/动态 micro-batch。

读完这两章后，你应该能沿着一条具体 trajectory 回答：它的 score 放在哪里、怎样变成 advantage、哪些 token 有梯度、PPO ratio 比较哪两个 policy、一次 trainer step 到底执行了多少次 optimizer update。
