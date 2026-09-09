# MiniCPM5–MQR–Sayuri 在线学习验证报告

> 日期：2026-08-24。本文记录机制实现、数学边界和单次初步实验；原始指标以 `analysis/results/minicpm_go_sayuri_*.json` 为准。

## 1. 结论

本仓库已经实现一条可运行的在线闭环：冻结 MiniCPM5-1B 主干提取棋盘语义特征，末层 `q_proj/v_proj` 挂接 FP32 LoRA，两个非线性 MQR 分别处理黑/白上下文，Sayuri 经 GTP 在模型预测后给出监督动作。每一步都先用参数 \((\phi_t,\psi_t)\) 预测，再由同一损失生成 MQR 隐式梯度和 LoRA 链式梯度，最后进入 \((\phi_{t+1},\psi_{t+1})\)。

初步结果证明了工程闭环、梯度连通、规则一致性、在线损失下降和约束稳定性；没有证明棋力、LoRA 增益、长期持续学习或“动物级学习”。在 48 步短流中，启用和禁用 LoRA 的结果几乎相同，当前可见学习主要来自 MQR 动作头。

## 2. 实际系统，而非概念图

- **主干**：`/home/spikebai/checkpoints/MiniCPM5-1B-AWQ-INT4`，24 层、隐藏维 1536。AWQ 4-bit 非对称分组权重按 `group_size=32` 校验并一次性解包为冻结 FP16 张量；这不是原生 INT4 反向训练。
- **LoRA**：只包装最后一个解码层的 `q_proj/v_proj`，rank 8、alpha 16，共 43,008 个可更新 FP32 参数；主干执行张量不更新。
- **MQR**：两个 64 节点、tanh 状态环按落子方显式 Top-1 路由，共 45,312 个可训练参数。完整 Cayley、注入和读出梯度先统一投影，再受原子 trust region 限制。
- **围棋**：`mqr/go.py` 实现 5×5 提子、禁自杀、情境超级劫、双 pass 与中国面积计分。合法动作掩码只用于选择可落子的模型动作；原始合法率另行报告，避免掩盖模型错误。
- **教师**：本地 `UsedCode/Sayuri` 固定在 commit `396b1d07e03a8a7ed1a2d806b3f895781ea7761c`。GPLv3 引擎独立构建并通过 GTP 子进程调用，没有复制或导入其源码。第 5 轮权重 SHA256 为 `31e31e6b8c3af59bc996470c91c262ea9aeb4fb1df6c8d2d5fafc320240bcb81`。

模型页面：<https://www.modelscope.cn/models/cyankiwi/MiniCPM5-1B-AWQ-INT4>。Sayuri 上游：<https://github.com/CGLemon/Sayuri>。网页说明用于核对架构和上游用法；量化布局以本地 checkpoint 元数据为准。

## 3. 外部 LoRA 梯度为何成立

令 LoRA 参数为 \(\phi\)，MQR 参数为 \(\psi\)，归一化后的主干特征为

\[
z(\phi)=\operatorname{LN}(f_{\theta_0,\phi}(p)),\qquad
\mathcal L(\phi,\psi)=\ell(R_\psi(z(\phi)),y),
\]

其中 \(\theta_0\) 冻结。MQR 的正向固定点和伴随固定点给出更新前参数处的

\[
g_z=\nabla_z\mathcal L(\phi_t,\psi_t).
\]

因此向特征图执行向量–Jacobian 乘积得到

\[
\nabla_\phi\mathcal L
=J_{z,\phi}^{\top}g_z,
\]

正是完整链式法则。实现虽然先提交 MQR、后提交 LoRA，但 LoRA 使用的是提交前缓存的 \(g_z\) 和特征计算图，所以两者等价于从同一 \((\phi_t,\psi_t)\) 计算的一阶 Jacobi 更新，而不是用新环参数污染旧梯度。

## 4. 分块正交更新与 trust region

将参数分为 MQR 与 LoRA 块。第 \(b\) 块的学习率预条件矩阵为 \(D_b\)，历史白化梯度基为 \(Q_b\)，更新为

\[
\Delta\theta_b=-c_bD_b^{1/2}(I-Q_bQ_b^\top)D_b^{1/2}g_b,
\qquad 0\le c_b\le1.
\]

若每个旧梯度块都被对应 \(Q_b\) 张成，则对任意独立 trust-region 缩放 \(c_b\)，

\[
g_{old}^{\top}\Delta\theta
=\sum_b g_{old,b}^{\top}\Delta\theta_b=0,
\]

而当前损失的一阶变化为

\[
g^{\top}\Delta\theta
=-\sum_b c_b\left\|(I-Q_bQ_b^\top)D_b^{1/2}g_b\right\|_2^2\le0.
\]

所以分块 OGD 仍给出整体一阶保护，而且比单个全局投影约束更强；代价是可能更快耗尽可塑性。保证只覆盖已成功存入有限秩基的方向，不覆盖二阶项、陈旧梯度、路由错误或记忆满载后的样本。

OGD 采样必须按上下文独立计数。若用全局 `step % 6` 对黑白交替流采样，会永久只记住一种颜色。当前实现已改为每个上下文各自执行 `ordinal % remember_every`，48 步对照中两个环的最终秩均为 4。

## 5. 初步实验

共同设置：seed 2026、5×5、Sayuri raw policy 教师、rank-8 LoRA、两个 MQR、OGD rank 8、每个上下文每 6 个样本记忆一次。Sayuri 与本地规则在 8 个抽查局面上合法着集合完全一致。

| 流 | LoRA 更新 | 首/末四分位 loss | held-out loss | 回放 masked 一致率 | 步时延 | 峰值 CUDA |
|---|---:|---:|---:|---:|---:|---:|
| 单盘 24 步 | 是 | 3.247 → 3.127 | 3.289 → 3.170 | 29.2% | 42.4 ms | 2.20 GB |
| 8 局面循环 48 步 | 是 | 3.149 → 2.249 | 3.289 → 3.125 | 75.0% | 37.4 ms | 2.20 GB |
| 8 局面循环 48 步 | 否 | 3.149 → 2.249 | 3.289 → 3.125 | 75.0% | 31.5 ms | 2.20 GB |

24 步流完成一局，最大酉误差为 `5.49e-6`。48 步 LoRA/no-LoRA 的逐步 loss 平均绝对差仅 `2.17e-5`，启用 LoRA 还增加约 5.9 ms/步；因此目前只能确认 LoRA 梯度链闭合，不能确认它带来统计或实用收益。8 个位置被重复六次，75% 回放一致率主要是小样本适应证据；held-out 只有 12 个位置和一个 seed，不能视作泛化结论。

### 后续多局结果

后续 3-seed 因果对局已经补齐了本报告的主要证据缺口。原生 Sayuri 反馈使固定 probe loss 改善 `0.335±0.067`，但实战目差退化 `11.5±8.2`，并发现策略向 pass 塌缩。pass 课程能改变该行为，但仍未证明规则内化或稳定棋力；正确规则也没有显著优于错误规则。以 [`minicpm_go_real_games_report.md`](minicpm_go_real_games_report.md) 为最新结论和复现入口。

在线 checkpoint 同时保存棋盘走子历史、环状态、路由、两个 OGD、上下文计数和累计步号。从 24 步状态恢复的 smoke run 正确输出 step `24,25`，计数从 `black=13, white=11` 连续到 `14,12`。旧格式缺少棋盘历史，加载时会主动清空瞬时环状态，避免把旧棋局状态拼接到空棋盘。

## 6. 合理性、创新性与不可行部分

合理之处是：冻结大主干控制显存；环的压缩映射和 Cayley 参数化给出可证明稳定性；显式环隔离和 OGD 分别提供结构级与局部一阶保护；GTP 教师使反馈发生在预测之后，适合真实在线协议。

组合创新候选是“精确隐式梯度的非线性 unistochastic sidecar + 多环上下文隔离 + 分块 OGD + 外部 LoRA VJP”。但各部件单独都不是新概念，且线性平衡环严格可合并为普通 LoRA。只有非线性、状态生命周期、路由或学习规则在等参数消融中带来优势，组合创新才有性能意义。

当前不可支持的结论包括：

- AWQ INT4 主干正在原生量化训练；实际是一次解包后冻结 FP16 执行。
- 该实验测得语言生成能力或围棋棋力；实际输出是外部 26 类动作头。
- 正交/酉参数自动消除遗忘；它只给有限方向的一阶局部保证。
- 已模拟一般动物学习；系统尚无主动探索、自监督世界模型、稀疏奖励归因、睡眠巩固和无任务 ID 的可靠路由。

## 7. 下一阶段可证伪研究

1. 用至少 5 个 seed、更多独立局面和固定计算预算比较 MQR-only、LoRA-only、MQR+LoRA、OGD-LoRA、replay 与线性头。
2. 把“重复局面记忆”升级为顺序任务：开局/中盘/劫争、教师策略切换和奖励反转；持续报告旧集合遗忘，而不是只报最终回放。
3. 分别比较 Sayuri raw policy 与固定 playout 的 MCTS 标签；保留 teacher strength 与 student adaptation 两套指标。
4. 将合法性作为训练约束或结构化动作空间消融，不能只依赖推理后掩码。
5. 扩到生成式接口前，加入冻结 LM head 的 token loss、延迟反馈队列和 KV-cache/环状态生命周期；在同预算下胜过 LoRA/replay 后再谈 2B。

## 8. 复现

```bash
bash scripts/setup_sayuri.sh
python3 test_go_minicpm.py
python3 experiments/minicpm_go_online.py \
  --teacher sayuri-policy --stream-mode game --rollout-policy teacher \
  --steps 24 --metrics-path analysis/results/minicpm_go_sayuri_game_lora_24.json
python3 experiments/minicpm_go_online.py \
  --teacher sayuri-policy --stream-mode dataset --steps 48 --dataset-size 8 \
  --metrics-path analysis/results/minicpm_go_sayuri_dataset_lora_48.json
python3 experiments/minicpm_go_online.py \
  --teacher sayuri-policy --stream-mode dataset --steps 48 --dataset-size 8 --no-lora \
  --metrics-path analysis/results/minicpm_go_sayuri_dataset_no_lora_48.json
```
