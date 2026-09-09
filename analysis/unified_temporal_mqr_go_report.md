# Unified Temporal Utility MQR Agent：设计与历史 3×3 机制验证

> 本报告保存原 3×3 核心隔离实验。3×3 不能作为围棋能力证据；修正 pass 联合
> 概率后的 10×10 诊断见
> [`mqr_10x10_sidecar_pilot_report.md`](mqr_10x10_sidecar_pilot_report.md)。

## 结论先行

统一 Agent、四头 Go 输出、通用上游梯度、双轨迹 write critic、离线 AWR、
短轨迹 actor-critic、慢速 MiniCPM LoRA，以及 identity/GRU/fast-weight 对照均已
实现并通过事务测试。**但 MQR 的独立竞争优势仍未成立。**最终 5-seed 门为
`mqr_effective=false`：参数与解析 MAC 已匹配到 5% 内，状态量未匹配；MQR 也
没有同时改善未掩码合法落子、真实回报、历史依赖合法性和跨任务遗忘。

权威结果为 `analysis/results/unified_temporal_mqr_go.json`，可由
`analysis/unified_temporal_mqr_go_verify.py` 独立重算。MiniCPM 链路结果为
`analysis/results/unified_temporal_mqr_minicpm_go_smoke.json`。

## 统一结构

对时间尺度 (j)，一次外部观测只推进一次状态：

\[
h_t^{(j)}=\tanh\!\left((1-\lambda_j)h_{t-1}^{(j)}H_j^\top
 + w_t^{(j)}J_j(x_t)\right),\qquad H_j=|U_j|^2,
\]

\[
U_j=(I-A_j)(I+A_j)^{-1},\qquad A_j^\dagger=-A_j.
\]

快环始终写入；utility gate 只控制慢环。若 (H_j) 双随机，则由
Birkhoff--von Neumann 分解有 \(\|H_j\|_2\le 1\)。因 `tanh` 为 1-Lipschitz，

\[
\|h_t-h'_t\|_2\le(1-\lambda_j)\|h_{t-1}-h'_{t-1}\|_2,
\]

所以状态稳定性保留。需要强调：酉性属于底层 (U)，真实状态传播使用的
(H=|U|^2) 一般**不正交且不保持能量**；OGD 的“梯度正交”又是另一条独立
约束。三者不能混称为“整体正交网络”。

Agent 输出 placement、legality、pass、value 四头。策略不使用规则 mask，而用
学习到的合法性软先验。当前实现先构造条件落点分布

\[
\tilde z_i=z_i^{place}+\beta\log\sigma(z_i^{legal}),
\]

再以二元 pass score $s$ 得到

\[
\log P(i)=\log\sigma(-s)+\log\operatorname{softmax}(\tilde z)_i,
\qquad \log P(pass)=\log\sigma(s).
\]

旧版直接拼接 pass 与 placement logits 的 10×10 结果不再使用；二者语义不同，
该拼接会使 pass 校准依赖棋盘面积。

非法点仍有有限概率，故 raw legality 可测。训练损失为

\[
L=L_{place}+\alpha_lL_{legal}+\alpha_pL_{pass}+\alpha_vL_{value}
 +\alpha_{ill}\sum_i\pi_i(1-y_i^{legal})+\alpha_{AWR}L_{AWR}
 +\alpha_{PPO}L_{PPO}-\alpha_H\mathcal H+\alpha_{KL}D_{KL}.
\]

其中 AWR 使用 \(-\exp(\hat A/\tau)\log\pi(a)\)，actor 阶段使用 clipped PPO；
actor 学习率为离线阶段的 (1/4)。长期策略 RL 未启用。

## 双轨迹效用的严格目标

票据保存当前动作后的 no-write/write 两个影子状态。两支接收相同未来局面，
后续慢写均关闭，只隔离当前写动作：

\[
R^b_t=\sum_{k=1}^{K}\gamma^{k-1}[-L_{t+k}^{(b)}],\qquad
A_t=R_t^1-R_t^0-c_m.
\]

在决策前因果特征 (c_t) 下，写入的条件期望增益是
\(\mathbb E[A_t\mid c_t]\)。因此 Bayes 最优动作严格为

\[
w_t^*=\mathbf 1[\mathbb E(A_t\mid c_t)>0].
\]

平方回归风险的唯一点态最优解满足

\[
g^*(c)=\arg\min_g\mathbb E[(g-A/s)^2\mid c]
=\mathbb E[A\mid c]/s,
\]

所以门取 `g(c)>0` 与最优动作一致。实现过程中发现旧设计逐样本裁剪 (A/s)。
这在数学上不保持均值符号：例如 (A=(10,-1,-1,-1,-1,-1)) 的均值为正，
裁剪到 ([-4,4]) 后均值为负。当前实现已移除裁剪，以原子参数信赖域控制异常值。

critic 使用长度 16 的因果特征缓冲、每 4 步两次小批量更新；阶段末冻结 task
head 后重新生成双轨迹目标并校准 20 步。utility 目标随 task head 改变，旧 critic
梯度不是固定旧损失，故不再对 utility 使用跨阶段 OGD；四个 task core 仍使用
相同 rank-4 OGD。这是 OGD 假设边界，不是取消 MQR 的正交记忆。

## 因果与可辨识性

每个事务严格执行：

1. 用 \(\theta_t\) 预测并执行 task/write 动作；
2. 保存不可变影子状态；
3. 环境产生未来轨迹和反馈；
4. 更新 critic、task core/head，得到 \(\theta_{t+1}\)。

当前标签不可能改变当前动作。`grad_logits` 接口允许固定点 MQR 接收任意已约简
的 \(\partial L/\partial logits\)，并复用隐式伴随、Cayley 拉回、OGD、裁剪和
原子提交；CE 与通用梯度事务已逐参数精确对齐。Temporal Agent 本身是单步动态，
使用截断 autograd，并以 `grad_features` 向外部 LoRA 传播，不能把两者混为同一个
隐式求解器。

普通完整棋盘几乎不需要时间记忆。为避免伪验证，实验加入超级劫配对任务：
ko 与 fresh 条件的当前棋盘、轮次、上一步和全部编码特征逐元素相同，但前者因
`position_history` 禁止立即回提，后者允许。对平衡配对，任何无状态函数
(f(x_t)) 在两条件输出相同，焦点合法率上限为 (1/2)；只有历史状态可能突破
该上限。训练只见 4 种棋盘对称，探针使用另 4 种。

## 等协议、近等参数/FLOP 对照

四方法共享编码器、路由、四头、损失、数据次序、OGD rank 和更新次数。

| core | 可训练参数 | 持久状态标量 | 解析前向 MAC |
|---|---:|---:|---:|
| MQR unistochastic | 8,371 | 9 | 8,464 |
| MQR identity | 8,406 | 42 | 8,466 |
| GRU | 8,724 | 12 | 8,124 |
| fast-weight | 8,673 | 75 | 8,220 |

参数比为 1.0422，MAC 比为 1.0421，均通过 5% 门；状态比为 8.3333，未通过
1.05 严格门。MQR 状态最少虽然是资源优势，但它同时造成表达容量不等，不能据此
宣布核心更优。2 维环还会退化为仅一个有效混合自由度，因此最终采用 3 维环。

## 历史 3×3、5-seed 结果

| 方法 | A 教师一致率 | raw 合法率 | useful 合法落子率 | 对局回报 | 劫争焦点合法率 | 正遗忘 |
|---|---:|---:|---:|---:|---:|---:|
| MQR | 0.3889 | 0.6500 | 0.6500 | -1.450 | 0.5000 | 0.0340 |
| identity | 0.4222 | 0.6500 | 0.6500 | -1.475 | 0.5222 | 0.0093 |
| GRU | 0.3944 | 0.6500 | 0.6444 | -1.525 | 0.5222 | 0.0491 |
| fast-weight | 0.4278 | 0.6333 | 0.6333 | -2.150 | 0.5333 | 0.0117 |

最终 raw pass rate 为 MQR/identity/fast-weight 的 0，GRU 为 0.0056，故表中合法率
不是全 pass 假象。配对 bootstrap 中，MQR 仅在回报上稳定超过 fast-weight：
`+0.70`, 95% CI `[0.20, 1.275]`。对 identity/GRU 的回报差分别为 `+0.025`
和 `+0.075`，区间均跨 0。MQR 对三个对照的劫争焦点差均非正；合法落子率和
遗忘差也全部未形成同时为正的置信区间。

MQR task-B critic 的在线符号准确率为 0.5567、平衡准确率为 0.5787，而多数符号
基线为 0.8433；平均 write-decision regret 为 0.02268。阶段末重新校准后，MQR
首观测真实/预测优势为 `-0.02493/-0.02425`，说明在当前已学 head 下保存上下文
平均有害，不是门遗漏了一个已存在的正优势。劫争焦点最终仍为机会水平 0.5。

因此当前证据支持“统一在线机制可运行”，否定“当前 MQR 核心已有独立优势”。

## MiniCPM 慢速 LoRA

本地 `/home/spikebai/checkpoints/MiniCPM5-1B-AWQ-INT4` 真实运行了 3 个快更新。
21,504 个 LoRA 参数共有 3 次机会，日程只触发 1 次更新；4 个适配器张量中 2 个
发生变化，最大漂移 `3.70696e-7`，冻结参数抽样漂移严格为 0，OGD rank 为 1。
但两样本 probe loss 从 `3.01385` 变为 `3.09376`，教师一致率仍为 0。它只证明
外部特征梯度和慢日程接通，不证明 LoRA 增益。该检查点标称 1B，不应写成 2B。

## 下一阶段必须满足的条件

1. 构造共享低秩注入、相同状态维数的 MQR/identity/GRU/fast-weight 核心，避免
   用不同隐藏容量换取参数/FLOP 匹配。
2. 给 utility gate 增加无标签的 candidate-impact 特征，例如
   \(\|\Phi(h,x,w=1)-\Phi(h,x,w=0)\|\)，并报告 held-out regret、AUPRC、平衡
   准确率和 write-rate，而非只看训练符号命中。
3. 正式实验至少使用 10×10，推荐 13×13、10–20 seeds，并扩展多种真实劫形
   与跨局历史；MQR 必须在未见劫形上以置信区间超过 0.5，并超过 identity 与 GRU。
4. 固定离线策略后再启用短 RL；当前模仿 loss 改善而对局回报下降，说明 surrogate
   与行为目标仍错位。长期 RL 现在只会增加归因混淆。
5. 只有核心门通过后，才做多种子 MiniCPM MQR-only/LoRA 对照和 Sayuri 外部评测；
   慢 LoRA 不能用来掩盖 MQR 核心失败。

即便这些门全部通过，也只能证明一种样本高效的持续学习 sidecar；“模拟一般动物
学习能力”还需要跨模态迁移、长期自主目标、能耗、开放世界适应和真实交互等独立
证据，当前实验远未覆盖。

## 复现

```bash
python3 test_unified_mqr_agent.py
python3 experiments/unified_temporal_mqr_go.py
python3 analysis/unified_temporal_mqr_go_verify.py
python3 experiments/unified_temporal_mqr_minicpm_go.py \
  --updates 3 --probe-examples 2 --lora-warmup 1 --lora-interval 2
python3 analysis/unified_temporal_mqr_minicpm_go_verify.py
```
