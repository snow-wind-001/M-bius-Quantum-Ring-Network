# Temporal MQR-v2：状态记忆与在线读出严格验证

> 日期：2026-08-24。权威结果为
> `analysis/results/temporal_mqr_delayed_digits.json`。这是三个 seed 的机制筛选，
> 不是 20-seed 确证实验，也不涉及 MiniCPM 或围棋。
>
> 后续受控写门、会话状态库、延迟反馈事务和真实 digit 干扰结果已经完成，见
> [`temporal_mqr_runtime_report.md`](temporal_mqr_runtime_report.md)。本文保留为
> 空白延迟基线，不用新结果回写旧实验口径。

## 1. 结论

本轮证明了时间型 MQR 可以同时实现可测的激活记忆和因果在线权重学习，
并发现“写入强度与遗忘速率解耦”是必要优化；但没有发现 Cayley-
unistochastic 拓扑或多时间尺度本身优于更简单对照。

- 800 个单遍在线序列后，480 个在线读出参数在 320 个未见延迟 Digits
  上达到 `77.50±1.13%`；相同初始化且完全不学习时为 `8.96±3.08%`。
- 固定点配置 `alpha=0.3, K=24` 的等价衰减控制为 `9.06±1.43%`。
  Temporal MQR 相对它的 seed-paired 改善为 `+68.44±2.25 pp`。
- 将写入强度从旧式 `kappa=lambda` 改为 `kappa=1`，在参数、随机初始化、
  转移和数据完全配对时改善 `+55.31±7.10 pp`。
- 单慢环为 `81.04±1.00%`，比多尺度环高 `3.54±0.18 pp`。该任务只有
  空白延迟，没有多时间尺度竞争或干扰，不能据此否定多尺度在其他任务的价值。
- identity 多尺度环为 `78.75±1.87%`，unistochastic 为 `77.50±1.13%`；
  paired 差值 `-1.25±0.83 pp`。当前证据明确不支持“Cayley 拓扑更优”。
- 所有 held-out 评估参数漂移精确为零；无学习条件训练漂移也精确为零。

## 2. MQR-v2 动力学

第 `j` 个环每个外部观察只更新一次：

\[
h_{j,t}=\sigma\!\left((1-\lambda_j)h_{j,t-1}H_j^\top
+\kappa_jJ_j(x_t)\right).
\]

`lambda` 控制遗忘，`kappa` 控制写入。当前实现支持 identity 或
`H=|U|^2`，后者由 Cayley 酉矩阵产生。冻结的 unistochastic 转移只求解
一次并缓存，避免逐 token 重复复数矩阵求解。

若 `sigma` 为 1-Lipschitz 且 `H` 双随机，则在无穷范数下

\[
\|h_t-h'_t\|_\infty
\le (1-\lambda)\|h_{t-1}-h'_{t-1}\|_\infty.
\]

因此状态半衰期严格为

\[
T_{1/2}=\frac{\log(1/2)}{\log(1-\lambda)}.
\]

写入/遗忘解耦不改变该收缩率。在线性无界激活下，若
`||J(x)|| <= M`，则

\[
\|h_t\|\le(1-\lambda)^t\|h_0\|
+\frac{\kappa M}{\lambda}\left(1-(1-\lambda)^t\right).
\]

所以强写慢忘会放大稳态上界；真实连续流应使用 tanh、归一化或受控写门，
不能无限制取大 `kappa/lambda`。

## 3. 因果实验

数据来自离线 `sklearn.datasets.load_digits`。每个序列是：

1. 只出现一次的 8×8 digit cue；
2. 1、4、16 或 32 个全零延迟帧；
3. 不含任何 digit 像素的 query；
4. 先用 `theta_t` 预测，再用标签更新读出到 `theta_(t+1)`。

训练/测试按 seed 分层切分，标准化统计只来自训练集。每个独立 digit
序列前清空状态。reservoir 条件冻结注入和转移，只更新 480 个读出参数，
没有 BPTT；因此延迟信息必须由环状态保存。测试还逐位验证 query 对不同
digit 完全相同。

## 4. 三 seed 结果

表中为 seed 均值 ± 样本标准差。

| 条件 | 在线参数 | prequential | held-out |
|---|---:|---:|---:|
| instant linear oracle | 660 | 85.88±0.94% | 93.13±0.83% |
| memoryless query linear | 660 | 9.38±0.75% | 10.10±1.18% |
| equilibrium K=24 control | 480 | 9.42±0.75% | 9.06±1.43% |
| single fast, lambda=0.3 | 480 | 29.54±1.58% | 34.79±2.22% |
| single slow, lambda=0.02 | 480 | 68.33±0.64% | **81.04±1.00%** |
| multiscale identity, strong write | 480 | 65.08±1.99% | 78.75±1.87% |
| multiscale unistochastic, kappa=lambda | 480 | 20.83±2.73% | 22.19±6.46% |
| multiscale unistochastic, kappa=1 | 480 | 64.54±1.92% | 77.50±1.13% |
| same multiscale, no learning | 0 | 9.00±1.15% | 8.96±3.08% |

快速环在 delay 1/4 上仍达到 `67.92/54.58%`，在 delay 16/32 上降至
`7.08/9.58%`，与其理论半衰期一致。慢环在四个 delay 上均保持约
`78–86%`，说明结果不是 query 泄漏或只利用短延迟样本。

## 5. 测试与证据边界

`test_temporal_mqr.py` 在本阶段建立的 14 项 v2 测试覆盖：精确指数衰减和半衰期、
unistochastic 收缩、写入/遗忘独立性、batch lane 隔离、冻结转移缓存、
拓扑和写入消融的逐位同初始化、query 无信息泄漏、preview 只读、反馈标签
隔离、状态/参数记忆归因、当前损失下降、全参数 OGD、全局裁剪及完整恢复。
v3 又增加 8 项运行时测试，当前总计 22 项；新增范围见后续运行时报告。

本轮没有证明：Cayley 拓扑优势、多尺度优于最优单尺度、延迟反馈下学习
注入层、抗干扰记忆、一般语言能力、围棋能力或动物级学习。三个 seed 只足以
筛选机制；强结论仍需开发 seed 与至少 20 个未触碰确认 seed。

## 6. 后续门槛状态

1. 已加入干扰 digit 和外部受控写门；oracle 门显著优于无门及同稀疏度随机门，
   但写门尚未学习；
2. 已实现有界 keyed state bank 和读出级延迟 feedback ticket；
3. 尚需 recent/remote 双查询、门控误差曲线和奖励反转，检验多尺度是否必要；
4. 用分层 feature replay 与功能 Jacobian 约束保护在线读出；
5. 通过上述门槛后，将 temporal state 接入分层围棋头，再接冻结 LM head。

## 7. 复现

```bash
python3 test_temporal_mqr.py
python3 analysis/temporal_mqr_result_verify.py
python3 experiments/temporal_mqr_delayed_digits.py
```
