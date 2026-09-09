# Temporal MQR-v4：因果可学习写门与实用在线运行时

> 日期：2026-08-24。学习门权威数据为
> `analysis/results/temporal_mqr_learned_gate_digits.json`，外部写门基线为
> `analysis/results/temporal_mqr_interference_digits.json`。本阶段验证的是
> **有即时辅助事件标签的在线门控**，不是无监督重要性发现、延迟奖励归因、
> 语言能力或棋力。
>
> 后续 v5 已实现“保存是否改善未来行为”的影子反事实收益门；请见
> [`utility_mqr_research_report.md`](utility_mqr_research_report.md)。本文继续作为
> v4 即时 marker 辅助门的权威记录，不用 v5 结果回写其历史结论。

## 1. 结论

Temporal MQR 现在同时维护三种可分辨记忆：环状态、任务参数和事件门参数。
当前帧严格先用 $(\theta_t,\phi_t)$ 选择写入并产生预测，随后任务标签和/或
事件标签才把参数更新为 $(\theta_{t+1},\phi_{t+1})$。任务梯度不能穿过写门；
门有独立 OGD、更新范数上限、计数器和 checkpoint 状态。外部门仍可覆盖学习门，
同时在后台训练控制器。

在 3-seed 真实 digit 干扰流中，44 参数的学习门将多尺度 held-out 准确率从
无门的 `19.86±1.68%` 提升到 `70.83±6.55%`，配对提升
`+50.97±6.74 pp`；不更新门、始终约 0.5 写入时只有
`21.53±0.87%`，门更新的配对效应为 `+49.31±6.47 pp`。oracle 为
`72.78±7.46%`，仅比学习门高 `1.94±1.58 pp`。

这个结果成立的关键限定是：控制器只接收一个显式、非类别的 cue marker，并在
每帧使用门后立即收到 cue/non-cue 辅助标签。它证明“事件门可因果在线学得并
保护 MQR 状态”，不证明模型能从稀疏任务奖励自主发现何时写入。

## 2. 运行时接口与事务顺序

核心状态更新为

\[
h_{j,t}=\sigma\!\left((1-\lambda_j)h_{j,t-1}H_j^\top
+\kappa_jg_{j,t}J_j(x_t)\right),\qquad g_{j,t}\in[0,1].
\]

学习门使用低秩控制器

\[
a_t=W_o\rho(W_de_t)+b,\quad p_t=\operatorname{sigmoid}(a_t/\tau),\quad
g_t=m+(1-m)p_t,
\]

其中 $e_t$ 可与 MQR 输入不同，$m$ 是最小写入量。默认不创建控制器，因而旧
代码仍为全开写入。提供 `gate_input_dim` 或 `gate_kwargs` 才启用学习门：

```python
learner = OnlineTemporalMQRClassifier(
    input_dim=65,
    ring_dim=12,
    output_dim=10,
    core_kwargs={"leak_rates": (0.5, 0.1, 0.02, 0.005)},
    gate_input_dim=1,
    gate_kwargs={"rank": 8, "minimum_gate": 0.02},
    gate_lr=0.05,
    gate_positive_weight=8.25,
    gate_max_update_norm=0.05,
)
result = learner.online_step(
    x,
    task_target,                    # 可为 None
    gate_input=event_feature,
    gate_target=event_label,        # 当前门使用后才监督
)
```

每步执行：

1. 用旧门参数计算 `learned_write_gate`；若有 `write_gate`，外部门成为实际门；
2. 将实际门 detach 后推进环状态并产生 `logits`；
3. 分别计算任务损失和辅助门 BCE，分别做完整向量 OGD 与范数裁剪；
4. 在事务尾部同时提交两组参数和旧策略产生的状态；只增加一次参数版本。

返回值显式区分 `write_gate_source`、`effective_write_gate`、
`learned_write_gate`、任务/门更新范数、门 BCE、false-open/false-close 和两套
OGD 统计。`online_gate_updates` 计更新事务，`online_gate_labels` 计展开到各
timescale 后的监督元素。`preview_step` 完全只读；`infer_step` 仍可立即提交
会话状态并签发单次 delayed-readout ticket。

## 3. 严格因果性与稳定性

实现把任务状态写成

\[
(\hat y_t,h_t)=F_{\theta_t}\!\left(x_t,h_{t-1},
\operatorname{stopgrad}(g_{\phi_t}(e_t))\right).
\]

事件标签 $c_t$ 不在右端。因此在输入、旧状态和旧参数相同时，仅改变 $c_t$，
当前 $g_t,h_t,\hat y_t$ 逐位相同；差异只能出现在更新后的 $\phi_{t+1}$。
同时
$\partial\ell_{task}/\partial\phi=0$，所以任务交叉熵不会偷偷把未来标签传入门。
27 项严格测试直接比较了两个只改变门标签的模型，并验证当前 gate/state/logits
完全相同、调用后门参数不同。

带正类权重 $w_+$ 的门损失为

\[
\ell_g=-\frac1N\sum_i\left[w_+c_i\log p_i
+(1-c_i)\log(1-p_i)\right].
\]

令 $z_i=a_i/\tau$，则

\[
\frac{\partial\ell_g}{\partial a_i}
=\frac{w_+c_i(p_i-1)+(1-c_i)p_i}{N\tau}.
\]

BCE 训练基础概率 $p$ 而不是带下限的 $g$，避免额外乘上 $(1-m)$；但 $m>0$
也意味着关闭门时仍有残余写入。门参数使用自己的白化 OGD 基。若
$w=\sqrt{\eta_g}\nabla_\phi\ell_g$、历史基为 $Q_g$，则

\[
\Delta\phi=-\sqrt{\eta_g}(I-Q_gQ_g^\top)w,\qquad
\nabla_\phi\ell_g^\top\Delta\phi
=-\|(I-Q_gQ_g^\top)w\|_2^2\le0.
\]

独立正标量裁剪不改变此一阶符号或对已保存方向的正交性。它仍只是局部一阶
保证，不是长期不遗忘定理。

若 `σ` 为 1-Lipschitz、`H` 双随机，令 $q=1-\lambda$，则

\[
\|h_t-h'_t\|_\infty\le q\|h_{t-1}-h'_{t-1}\|_\infty.
\]

对 oracle 门 $g^*$ 与近似门 $g$，若 $\|J(x_s)\|_\infty\le M$，则

\[
\|h_t-h_t^*\|_\infty
\le q^t\|h_0-h_0^*\|_\infty
+\kappa M\sum_{s=1}^{t}q^{t-s}|g_s-g_s^*|.
\]

所以误开会反复注入干扰，漏写会削弱 cue；二者影响还取决于发生时间，不能只用
未加权错误率替代任务评估。

## 4. 外部写门基线

第一组实验没有事件 marker，也不学习门。序列为一个目标 digit、1/4/8/16 个
真实 digit 干扰和全零 query；仅 query 先评分再更新 480 参数读出。

| 条件 | held-out |
|---|---:|
| memoryless zero-query linear | 10.21±1.54% |
| multiscale ungated | 20.10±1.83% |
| multiscale random one-write | 18.13±1.13% |
| multiscale oracle cue-only | 73.44±5.16% |
| single slow oracle cue-only | **76.25±0.94%** |

oracle 相对无门配对提升 `+53.33±6.17 pp`，证明可靠选择性写入有价值；随机
同稀疏门无效，证明收益不只是“少写”。

## 5. 学习门实验

第二组实验在每帧增加一个 marker：cue 为 1，digit 类别信息仍只在 64 个像素
中；干扰和全零 query 的 marker 均为 0。控制器**只看这个 marker**，并在门已
使用后收到 cue/non-cue 标签。任务类别仍只在 query 预测后到达。训练 600 条、
held-out 240 条；所有条件逐 seed 初始参数 SHA-256、数据、顺序、干扰和腐败
uniform 完全配对。reservoir/注入冻结；held-out 任务和门参数漂移精确为零。
正类权重 `8.25` 等于平衡的 1/4/8/16 干扰协议中每个 cue 对应的平均
`distractor + query` 负事件数，不是按 held-out 结果调出的阈值。

| 条件 | held-out | 在线参数 |
|---|---:|---:|
| multiscale ungated | 19.86±1.68% | 480+44 |
| multiscale oracle | 72.78±7.46% | 480+44 |
| multiscale learned gate | **70.83±6.55%** | 480+44 |
| learned gate, no gate updates | 21.53±0.87% | 480 |
| oracle + 10% false-open | 52.50±3.56% | 480+44 |
| oracle + 25% false-open | 39.03±4.88% | 480+44 |
| oracle + 10% false-close | 65.42±8.21% | 480+44 |
| single slow learned gate | 73.61±5.36% | 480+17 |

学习后的多尺度控制器 held-out 二值准确率为 `100%`，cue 平均门值
`0.9953±0.0001`，非 cue 为 `0.04249±0.00034`。相同控制器在 external
gate 条件后台训练时最终参数 hash 逐 seed 精确相同，证明其梯度不依赖任务门
路径。10%/25% 实际误开率为 `10.07±0.85%`/`24.70±1.46%`，相对 oracle
任务准确率分别下降 `20.28±5.37 pp`/`33.75±7.23 pp`；约 10.28% 漏写造成
`7.36±0.87 pp` 下降。在此协议中，误开比同率漏写更昂贵。

多尺度学习门仍比等状态单慢环低 `2.78±1.20 pp`。因此新增证据支持学习门，
仍不支持多尺度或 Cayley 拓扑优越性。

## 6. 可行性与仍未解决的问题

这版已具备 sidecar 的必要工程语义：有界 keyed state、选择性写入、同步双损失
事务、独立 OGD/裁剪、外部安全覆盖、延迟读出票据和完整恢复。44 个门参数的
成本很小，适合作为冻结 1B/2B 主干外部的事件控制器。

但从本实验到“模型自主在线学习”仍有明确鸿沟：

1. marker 和即时事件标签由协议提供；稀疏奖励下何时写的信用分配尚不存在；
2. 门只看当前事件特征，没有使用预测误差、状态新颖性、不确定性或长期回报；
3. 单慢环继续胜过多尺度；需 recent/remote 双查询证明时间尺度分工；
4. 门标签类不平衡由人工 `gate_positive_weight` 补偿，真实流需在线估计先验并
   防止全关/全开；
5. ticket 仍只更新读出；LM LoRA/注入的延迟更新需要参数快照或 feature replay；
6. Python runtime 需要 learner 级串行提交；它不是无锁并发参数服务器。

面向 MiniCPM/2B 的下一步应使用冻结 LM 隐状态中的显式事件特征作为 $e_t$，先
做一次性事实写入、上下文切换和奖励反转，并与同参数 GRU、fast-weight、LoRA、
replay 比较。只有无显式 marker 时仍能在 held-out 流上学得可靠门，且旧任务
退化、P95 延迟和显存优于基线，才可讨论更一般的持续学习能力。

## 7. 复现

```bash
python3 test_temporal_mqr.py
python3 experiments/temporal_mqr_interference_digits.py
python3 analysis/temporal_mqr_interference_verify.py
python3 experiments/temporal_mqr_learned_gate_digits.py
python3 analysis/temporal_mqr_learned_gate_verify.py
```

结论边界：本阶段严格证明并验证了“使用当前事件特征、在当前决策后收到辅助
事件标签时，低秩 MQR 写门可在线学得并显著减少干扰”。它没有证明奖励驱动的
自主门控、多尺度优势、2B 语言能力提升、围棋实战提升或一般动物学习。
