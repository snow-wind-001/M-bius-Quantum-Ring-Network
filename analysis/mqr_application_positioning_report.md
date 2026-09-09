# MQR 冻结小模型 Sidecar：应用定位核实与测试报告

## 1. 核实结论

“冻结小模型的受控在线个性化、会话记忆和错误修正层”是当前最合理的应用定位，
但应称为**已通过机制资格验证的研究/工程目标**，不能称为已经证明有效的真实模型
产品。现有证据不支持“通用能力增强器”“动物式学习模型”或“MQR 已具有独立
算法优势”。

原因是三层证据必须分开：

1. 结构与事务层已经成立：初始零扰动、predict-before-update、会话状态隔离、
   有符号状态界、residual 限幅和候选回滚均可测试；
2. 冻结主干应用机制在受控合成任务上成立，MiniCPM5-1B 链路也能保持主干 probe
   零漂移；
3. 真实任务增益和 MQR 相对其他侧车的独特优势仍未成立。

## 2. 新增应用资格实验

权威结果为
`results/frozen_sidecar_application_qualification.json`。实验使用 10 个固定 seed，
冻结一个小型特征主干和原生分类头，只在线更新 MQR residual readout。两个会话在
相同 query 上要求相反标签，只有先前 profile token 和各自环状态可区分它们。
每个 seed 接收 100 次反馈，严格先预测后更新。一个预注册旧域 probe 通过读出梯度
的精确零空间投影保护；Cayley/注入保持冻结，转移矩阵缓存。

| 检查 | 结果 | 解释边界 |
|---|---:|---|
| 初始 sidecar 输出漂移 | `0` | 初始严格 no-op |
| 冻结主干/原生头参数漂移 | `0` | 更新未进入主模型 |
| 固定 sidecar 非读出参数/转移漂移 | `0 / 0` | 只有读出在线变化 |
| 冻结主干个性化准确率 | `0.500` | 同 query 无法表达冲突偏好 |
| 保留会话状态的 MQR 准确率 | `1.000`，10/10 seeds | 合成任务上能使用上下文状态 |
| query 前清空状态 | `0.500` | 改善确实依赖会话记忆 |
| 被纠正会话 B 的平均 NLL | `0.9038 → 0.2780` | 多次反馈后定向错误得到修正 |
| 受保护旧域最大 policy KL | `2.86e-16` | 仅对声明的单个 probe 近机器精度守恒 |
| residual 最大 L2 | `2.0` | 满足预注册限幅 |
| signed-state 证书 | `10/10` | 本次轨迹均满足解析界 |
| 故意越界候选 | `10/10` 拒绝并零漂移回滚 | 证明应用层事务实现正确 |

独立 counterfactual 测试给同一旧参数/输入提供相反标签，两个 issue-time logits
差为 `0`，更新后才产生 `0.0425` 的差异。交错执行两个 stream 与分别执行的
logits/state 最大误差也均为 `0`。因此标签泄漏和会话串扰在所测路径上被排除。

该实验不证明 MQR 特有优势：固定 reservoir 加线性读出也可能由 ESN、GRU、LSTM
或 fast-weight 实现。它还暴露一个工程边界：通用 `TemporalMQRSidecar` 提供诊断，
但候选提交事务仍由调用方持有；Go Agent 已内建事务，通用冻结模型包装器尚未统一。

## 3. 真实主干与围棋证据复核

MiniCPM5-1B smoke 的冻结参数 probe 漂移仍为 `0`，且遵守
predict-before-update；这只证明接口连通和冻结语义，不证明个性化、纠错、语言
质量或围棋能力提升。

正式竞争实验现已完成 10×10 主实验和 13×13 同协议复验。每个尺度使用 10 个
预注册 seed、每任务 128 次主在线反馈、64 个 held-out probe 和每阶段 8 局无合法性
mask 的完整对局，并包含 identity、GRU、LSTM、fast-weight、LoRA、OGD-LoRA、
equal-byte replay-LoRA 与隐式 Sinkhorn。所有方法的循环状态均为 48 B，两个尺度的
参数、前向 MAC 和分配更新 MAC 最大比均低于 `1.05`。

MQR 的 task-A held-out loss 在 10×10 和 13×13 分别降低 `2.8331` 和 `3.0168`，
因此系统确实执行了在线权重学习。但是两尺度所有方法的历史焦点准确率均为
`0.500`，MQR 在核心指标上没有超过 identity、LSTM、fast-weight 或 Sinkhorn。
13×13 上 replay-LoRA 的动态有效合法率和回报还显著优于 MQR：按 MQR−replay
计算分别为 `-0.00407 [-0.00633,-0.00136]` 和
`-0.1125 [-0.1750,-0.0375]`。联合验证因此保持
`mqr_effective=false`。这不是实验缺失或旧状态资源不公平导致的未知结论，而是
正式协议下的负竞争结果。

## 4. 正式条件完成度与剩余性能缺口

| 正式要求 | 当前状态 | 是否满足 |
|---|---|---:|
| 10–20 seeds | 10×10/13×13 各 10 seeds | 是 |
| 至少 10×10 正式在线训练 | 每任务 128 主反馈、64 probe、8 局动态评估 | 是 |
| 13×13 规模扩展复验 | 同协议正式结果与联合审计 | 是 |
| identity、GRU、LSTM、fast-weight | 同路由、损失和预算 | 是 |
| LoRA、OGD-LoRA、equal-byte replay、Sinkhorn | 同大棋盘协议已接入 | 是 |
| 同参数、状态字节、FLOP | 状态比 `1.000`，其余最大比小于 `1.05` | 是 |
| 同反馈与梯度样本预算 | 每方法/seed 为 256 个唯一反馈、384 个梯度样本 | 是 |
| 合法率、回报、历史与遗忘联合胜出 | 联合有效性门为 `false` | **否** |

这里的“正式在线训练”指本项目预注册的 128 次/任务预算，不等同于大规模围棋
强化学习或基础模型长期训练。10 seeds 满足当前最低执行门，也不自动保证任意小
效应的统计功效。当前停止条件已从“补齐实验”转为“修复机制”：先在 marker-free
任务上降低干扰写率、建立反转学习，并使 identity replacement 产生显著性能损失；
这些条件未满足前，继续扩大棋盘或训练预算不能证明 MQR 特异性。

## 5. 应用前景与下一门槛

近期开发表述可以是：MQR 是一个小维、可缓存、带会话状态和安全预算的冻结模型
侧车，适合本地偏好、短期事实记忆、重复错误纠正和受控回滚。最实用的初版应冻结
Cayley 转移，只更新读出或低秩注入，从而避免每 token 的稠密 Cayley 求解。

部署前仍必须在真实冻结小模型上做同资源对照，指标至少包括 prequential regret、
纠错所需反馈数、会话间串扰、受保护集 KL/准确率下降、遗忘、P50/P95 延迟、峰值
内存、总状态字节和拒绝更新率。只有 MQR 在这些指标上同时超过 identity 与一个
非 MQR 核心，且不能由额外状态、反馈或保护 probe 解释，才可把“应用机制可行”
升级为“MQR 具有独立竞争力”。

## 6. 复现

```bash
python3 experiments/frozen_sidecar_application_qualification.py
python3 analysis/mqr_application_positioning_verify.py
python3 analysis/mqr_sidecar_safety_verify.py
python3 analysis/unified_temporal_mqr_go_competitive_verify.py \
  --require-replication \
  analysis/results/unified_temporal_mqr_go_competitive_10x10_formal_10seed.json \
  analysis/results/unified_temporal_mqr_go_competitive_13x13_formal_10seed.json
```

组合审计结果写入 `results/mqr_application_positioning_audit.json`。该审计明确保持
`real_model_application_gain=false`、`general_capability_enhancement=false`、
`animal_like_learning=false` 和 `independent_mqr_algorithm_advantage=false`。
