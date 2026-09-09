# 莫比乌斯量子环形网络 (Möbius Quantum Ring Network)

本 PyTorch 研究实现源自 HTML 中的 **MQR / UHR-Net (Möbius Quantum Ring / Unistochastic Hamiltonian Ring)** 设想，保留固定点推理路径：**Cayley 酉矩阵参数化 → 幺模随机连接 \(H=|U|^2\) → 固定点松弛推理 → LoRA 式注入 → 局部采样读出**；另提供跨观察的时间环记忆。当前探索新增直接保留符号的正交 Givens 环、轨迹反馈与 OGD 在线更新。历史公式中的数学错误已在当前代码中修正。

> **当前进展（2026-09-09）**：按新的研究要求恢复围棋在线探索，保留正交、环形结构与在线更新。
> 已实现 `GoOnlineSession`、有符号正交转移、当前参数下的 OGD 记忆重建、可选空间调节及对局保存恢复。
> 五种子对照中，OGD 使全局/空间版本的旧任务 NLL 增幅降低约 50% / 61%，同时减弱新任务适应；
> 正交环尚未超过 identity 对照，也未建立棋力优势。8 局真实在线运行完成 268 次反馈、38 次更新，全部告负。
> 详见 [完整分析、实测表与复现命令](analysis/orthogonal_online_go_report.md)。本轮测试对象是微型围棋网络，
> 不代表 MiniCPM 通用能力提升。使用 [CodeRecoder 入口](scripts/code_protect.mjs) 创建并独立校验外部保护快照。

最新 [技术根因分析与解决队列](analysis/current_technical_root_causes.md) 补充了 40 个 checkpoint 的只读诊断：空间版本的空棋盘位置别名、记忆视角逐手翻转、较弱的环转移梯度及 OGD 锚点漂移。区分已证实缺陷与待验证改进，保留正交、环形网络和在线更新的核心约束。

> **历史证据（2026-09-01）**：结构约束、非线性隐式梯度、有限求解证书和
> 原子在线事务均有自动化检查。正式 10×10/13×13 实验各用 10 seeds，已实现
> 48 B 循环状态接口和小于 1.05 的参数/前向/更新资源比。MQR 的 held-out
> loss 在两个尺度都显著下降，但历史焦点准确率仍为 `0.500`，且 13×13
> 的 replay-LoRA 在动态合法率与回报上显著优于 MQR。因此当前证据支持“可审计的
> 在线 sidecar 机制”，不支持“优于 LoRA/RNN 的独立算法”。
> 新版地址/内容分离路由已在 10-seed 合成任务上通过机制门，但与直接固定环形
> 缓冲区严格持平。随后完成的 Phase IV 使用未知 A/B 拓扑和严格匹配的
> event/distractor `(key,value)` 边缘分布；Givens/分块 Cayley 均恢复 100%
> 拓扑并击败 identity 与直接 ring buffer，但同资源可学习稀疏置换也达到
> `1.000`，故独立优势仍为 false。当时据此暂停了 Go/MiniCPM/策略 RL；本轮重新开放围棋研究，保留历史负面结论。

## 🌟 核心机制与研究假设

### 网络架构 (Architecture)

- **幺模随机约束 (Unistochastic Constraint)**：学习一个酉矩阵 \(U\in U(N)\)，并用 \(H=|U|^2\) 作为连接矩阵；\(H\) 天然双随机，但本身一般不正交，也不普遍保持 \(L_2\) 能量。
- **无头无尾环形动力学 (Headless Ring Dynamics)**：推理不是“前馈计算图”，而是通过固定点松弛迭代收敛到唯一稳态 \(h^\*\)。
- **哈密顿注入/LoRA 注入 (Hamiltonian / LoRA Injection)**：输入通过低秩注入算子驱动环形系统：\(\mathcal{J}(x)=W_{up}W_{down}x\)。
- **非线性扩展（可选）**：支持在注入端加入激活 \(\mathcal{J}(x)=W_{up}\,\sigma(W_{down}x)\)，以及在环内松弛加入 1-Lipschitz 激活 \(h\leftarrow\sigma((1-\alpha)hH^T+\alpha\mathcal{J}(x))\)（保持收敛性证明成立）。
- **Patch Embedding 前端（可选）**：支持 `image_encoder=patch`，用 stride=\(p\) 的卷积做局部特征提取，再 pool/flatten 得到注入向量（引入局部感受野与层次化表征）。
- **自保持混合 (H\_eff, optional)**：支持 \(H_{eff}=(1-\beta)I+\beta H\) 以抑制双随机传播的“过度平均化”，并保持双随机与收敛性不变。
- **复数酉推理（可选）**：支持 `dynamics_mode=unitary`，推理环用 \(U^\dagger\) 在复数域传播，读出时对 \(h^\*\) 做测量（默认 \(|h^\*|\)）以让相位参与推理。
- **局部投影采样读出 (Local Projective Sampling)**：只采样环上局部节点集合 \(\mathcal{S}\)（默认前 k 个节点）得到输出：\(y=W_{readout}\cdot h^\*_{\mathcal{S}}\)。
- **双酉矩阵解耦 (Dual-Unitary World/Policy, optional)**：将连接酉矩阵分解为 \(U_{total}=U_{policy}\,U_{base}\)，其中 \(U_{base}\) 冻结作为稳定“世界模型”，\(U_{policy}\) 可学习作为“策略”。
- **可学习GT平衡态 (Learnable Goal Equilibrium, optional)**：引入按类/按任务的可学习目标平衡态 \(p_y\)，用于状态级目标对齐（模拟“受教育/反思”过程）。
- **原型距离读出 (Prototype Readout, optional)**：可选用 \(\hat{y}_{c}=-\|h^\*_{\mathcal{S}}-p_{c,\mathcal{S}}\|^2/(2\tau)\) 直接产生 logits，将分类目标与目标平衡态绑定。
- **多环在线学习 (Online Multi-Ring)**：严格采用“先预测、后更新”的 prequential 协议；显式上下文隔离不同环，并可在按学习率白化的完整参数梯度上执行低秩 OGD。
- **时间型多尺度 MQR (Temporal MQR-v5)**：每个外部观察只推进一次环状态，显式分离遗忘率 `lambda`、写入强度 `kappa` 与门 `g`；v4 支持即时辅助事件门，v5 另以成对影子轨迹估计未来 write/no-write 损失差，让门学习“保存是否改善未来行为”，并支持候选延期晋升、独立 OGD/裁剪和完整恢复。
- **有符号正交时间环**：`transition_mode="orthogonal"` 配合 `transition_structure="cyclic_givens"`，直接稀疏施加相邻旋转，保持无阻尼传播的 L2 范数；阻尼后有明确历史衰减。它与 `H=|U|²` 的平均化传播不同。
- **围棋在线会话**：`GoOnlineSession` 支持先预测后反馈、有限轨迹梯度、未完成票据保存恢复；`consolidate_task_memory` 用已见训练窗口重建 OGD，`SpatialRingGoAgent` 可用环状态调节冻结空间特征。
- **残差地址路由 MQR**：内容保存在物理槽中且无写时逐位不变；
  (H_\varepsilon=(1-\varepsilon)I+\varepsilon T) 只产生地址路由分数，
  straight-through top-1 提交离散地址。写门具有硬预算、正负收益平衡、
  多时间尺度 utility 与 AUPRC/Brier/ECE 审计。
- **统一 Temporal Utility Agent**：共享因果 write critic，拆分 placement/legality/pass/value 四头，支持 AWR、短轨迹 actor-critic、通用上游梯度和慢速 LoRA 巩固；所有结论由 identity/GRU/fast-weight 严格门约束。
- **可认证固定点事务**：可选 tolerance 返回正向/伴随 residual、`residual/alpha`
  误差界和收敛状态；认证失败默认整笔拒绝权重、OGD 和记忆外梯度更新。
- **安全残差 Sidecar**：`TemporalMQRSidecar` 以零读出保证初始化时对冻结隐藏层
  严格无扰动，报告有符号状态上界和 residual 范数；统一 Agent 与 MiniCPM LoRA
  均可先执行候选更新，再按 KL、输出、状态和 Cayley 漂移预算提交或原子回滚。

### 算法改进 (What changed in this repo)

- **从“类ViT堆叠结构”重构为 MQR 固定点环**：历史结构来自 `Möbius Quantum Ring.html`；修正后的梯度、在线协议和结论边界以 [`analysis/mqr_math_proof.md`](analysis/mqr_math_proof.md) 为准。
- **在不破坏“严格HTML默认配置”的前提下补齐表达能力选项**：新增可选 patch embedding、\(H_{eff}\) 自保持混合、复数酉推理（measurement 读出）与 1-Lipschitz 非线性开关，用于在 CIFAR-100 等任务上探索“稳定性约束 vs 表达能力”的平衡。

### 1. **酉矩阵参数化 (Unitary Matrix Parameterization)**
- 使用 Cayley 变换从 **反埃尔米特矩阵 (skew-Hermitian)** 生成酉矩阵：`U = (I - A)(I + A)^(-1)`
- 保证 `U^† U = I`
- `coordinate_mode="minimal"` 以严格 \(N^2\) 个实坐标覆盖
  \(\mathfrak u(N)\)；旧 `projected` 模式存储 \(2N^2\) 个标量，仅为 checkpoint
  兼容保留
- Cayley 覆盖谱不含 \(-1\) 的开稠密酉子集；它省去 Sinkhorn 前向归一化循环，
  但引入 \(O(N^3)\) 稠密求解，不构成无条件速度优势
- 单个 Cayley chart 的 \(-1\) 谱遗漏不会缩小可表示的 unistochastic
  转移集：对任意酉代表 \(V\)，总能选取全局相位 \(\zeta\) 使
  \(-1\notin\operatorname{spec}(\zeta V)\)，且 \(|\zeta V|^2=|V|^2\)。这只解决全局
  表示覆盖，不消除 \(U\mapsto|U|^2\) 的局部 Jacobian 退化

### 2. **双随机权重自动生成**
- 由酉矩阵的行/列归一性可严格推出其逐元素模平方为双随机矩阵：`H = |U|²`
- 自动满足:
  - `H_ij ≥ 0` (非负性)
  - `∑_j H_ij = 1` (行和为1)
  - `∑_i H_ij = 1` (列和为1)
- 保证在 \(L_1/L_\infty\) 诱导范数下不扩张；阻尼后的固定点环严格收敛，但这不等于普遍的能量守恒或自动防止所有梯度消失

### 3. **推理环：固定点松弛 (Inference = Relaxation to Fixed Point)**
- 推理环(实数域)通过迭代收敛到稳态：
  - \(h \leftarrow (1-\alpha)\,h\,H^T + \alpha\,\mathcal{J}(x)\)
- \(\alpha\in(0,1]\) 是耗散/注入平衡系数，保证收敛稳定（Banach 不动点思想）。

### 4. **更新环：精确隐式梯度**
- `eqprop_update_step` 先求正向与伴随固定点，再通过 \(H=|U|^2\) 和 Cayley 微分做精确链式拉回；有限迭代是唯一的固定点求解近似。
- 工程同时提供诊断工具：
  - `MoebiusQuantumRing.compute_adjoint_state(...)`
  - `MoebiusQuantumRing.approx_grad_H(...)`
- 非线性状态的精确缩放伴随满足
  \(p=\alpha g(I-(1-\alpha)DH)^{-1}\)；线性闭式解不能在 \(D\ne I\) 时直接复用
- `mqr/baselines.py` 提供收敛 Sinkhorn 的隐式梯度，用于同等数学口径的对拍

## 📁 项目结构

```
MöbiusQuantumRing/
├── mqr/                          # 工程化核心实现（按HTML算法复现）
│   ├── unitary.py                 # CayleyUnistochasticParam (U, H=|U|^2)
│   ├── ring.py                    # MoebiusQuantumRing / injection / readout
│   ├── baselines.py               # 收敛 Sinkhorn 与隐式梯度公平基线
│   ├── online.py                  # prequential multi-ring / OGD memory
│   ├── temporal.py                # K=1 多时间尺度状态与在线更新
│   ├── routing.py                 # 内容/地址分离、残差路由与预算化 utility
│   ├── safety.py                  # constancy 预算、KL/漂移与 residual 诊断
│   ├── utility.py                 # 延迟反事实收益门、候选票据与慢环晋升
│   ├── agent.py                   # 统一四头 Agent、AWR/PPO、慢 LoRA 日程
│   ├── agent_baselines.py         # identity/GRU/fast-weight 与资源审计
│   ├── go_agent.py                # Go 向量/MiniCPM 编码、劫争配对与双轨迹
│   ├── go_online.py               # 因果在线会话、空间调节与 OGD 记忆重建
│   ├── minicpm.py                 # 本地 AWQ 解包、冻结主干与末层 LoRA
│   ├── go.py                      # 任意棋盘 Go 规则与基础数据集
│   ├── sayuri.py                  # 独立 Sayuri GTP 子进程适配
│   └── __init__.py
├── experiments/online_digits_principle.py # 无下载 Digits A→B 原理实验
├── experiments/temporal_mqr_delayed_digits.py # 延迟 Digits 状态记忆实验
├── experiments/temporal_mqr_interference_digits.py # 真实干扰/受控写门实验
├── experiments/temporal_mqr_learned_gate_digits.py # 因果辅助学习门实验
├── experiments/temporal_mqr_utility_digits.py # 无 marker 未来收益门实验
├── experiments/minicpm_go_online.py       # MiniCPM/MQR/Sayuri 在线闭环
├── experiments/orthogonal_go_online.py    # 小型 Go 网络的正交环与 OGD 对照
├── experiments/play_online_go.py          # 实际对局中继续学习与恢复
├── experiments/mqr_routed_memory_qualification.py # 无 marker 地址路由资格实验
├── experiments/mqr_contextual_topology_qualification.py # 未知上下文拓扑正式门
├── analysis/                     # 严格证明、研究方案与实验结果
├── scripts/code_protect.mjs        # 使用本地 CodeRecoder 内核建立校验快照
├── mobius_quantum_ring.py         # 向后兼容门面（re-export mqr/）
├── train_mobius_cifar100.py       # CIFAR-100训练脚本
├── quick_start.py                 # 快速开始示例
├── test_mobius_model.py           # 测试套件
├── test_online_learning.py         # 在线协议/隔离/OGD 测试
├── test_temporal_mqr.py             # 时间记忆、归因与恢复测试
├── test_utility_mqr.py              # 收益归因、晋升、OGD 与恢复测试
├── test_unified_mqr_agent.py        # 四头概率、资源门与统一 Agent 测试
├── test_go_minicpm.py              # AWQ/LoRA/围棋/Sayuri 集成测试
├── test_routed_memory.py            # 路由不变量、成本、预算与校准测试
├── README.md                       # 本文档
└── Möbius Quantum Ring.html       # 架构设计文档
```

## 🚀 快速开始

### 安装依赖

```bash
pip install torch torchvision numpy tensorboard
# 仅 Digits 原理实验需要
pip install scikit-learn
```

### 围棋在线探索

以下命令训练微型空间基网络，然后在正交环和 OGD 系统上继续学习；默认教师在本地运行。

```bash
# 需要相邻 CodeRecoder 工程已构建，也可用 --coderecoder-root 指定位置。
node scripts/code_protect.mjs snapshot --name before-online-go

python3 experiments/orthogonal_go_online.py \
  --seeds 17 --methods orthogonal_ogd identity_ogd orthogonal \
  --checkpoint-dir checkpoints/orthogonal_go_online \
  --output analysis/results/my_online_go.json

python3 experiments/play_online_go.py \
  --checkpoint checkpoints/orthogonal_go_online/orthogonal_ogd-17.pt \
  --games 8 --save-to checkpoints/live_go.pt

# 继续下一局，恢复模型、环状态与正交梯度记忆。
python3 experiments/play_online_go.py \
  --checkpoint checkpoints/live_go.pt --games 8 --save-to checkpoints/live_go.pt
```

这是教师监督的在线学习，默认每 8 次观察更新一次。真实对局在双方轮次都获得教师反馈；执行动作使用环境合法性约束，原始模型合法率另行记录。可选 `--readout spatial` 的独立实验、五种子结果及完整检查命令见 [研究报告](analysis/orthogonal_online_go_report.md)。checkpoint 保存在本地，不提交到 Git。

### 通用在线推理与学习

```python
import torch
from mqr import OnlineMultiRingClassifier

learner = OnlineMultiRingClassifier(
    input_dim=64, hidden_dim=48, output_dim=10, num_rings=2,
    lr=0.08, ogd_max_rank=16,
    ring_kwargs={"alpha": 0.35, "relaxation_steps": 12,
                 "lora_rank": 16, "readout_dim": 48},
)
x, target = torch.randn(1, 64), torch.tensor([3])
result = learner.online_step(x, target, context_id="context-a")
# result["logits"] 来自更新前 theta_t；调用结束后参数为 theta_(t+1)
```

固定点 sidecar 可启用求解认证和初始零扰动：

```python
from mqr import MoebiusQuantumRing

ring = MoebiusQuantumRing(
    input_dim=64, hidden_dim=16, output_dim=10,
    alpha=0.1, relaxation_steps=300, relaxation_tol=1e-8,
    relaxation_min_steps=4, lora_rank=8, readout_dim=16,
    state_activation="tanh", cayley_coordinate_mode="minimal",
    zero_init_readout=True,
)
update = ring.eqprop_update_step(
    torch.randn(2, 64), torch.tensor([1, 2]),
    lr=1e-2, adjoint_steps=300,
)
assert update["did_update"] == bool(update["solver_converged"])
```

认证开启时，`forward_residual`、`forward_error_bound`、
`adjoint_residual`、`adjoint_error_bound` 和迭代数随事务返回。未收敛默认
`did_update=False`，并保持参数、OGD 记忆和 `grad_x` 不变；
`allow_inexact_update=True` 只用于显式研究有限求解偏差。

冻结模型的最终隐藏表示可以通过零扰动时间型 sidecar 修正：

```python
import torch
from mqr import SidecarSafetyLimits, TemporalMQRSidecar

sidecar = TemporalMQRSidecar(
    hidden_dim=1536,
    ring_dim=64,
    residual_clip_l2=0.25,
    core_kwargs={
        "leak_rates": (1.0, 0.1, 0.02),
        "injection_rank": 8,
        "cayley_coordinate_mode": "minimal",
    },
)
frozen_hidden = torch.randn(1, 1536)
adapted, state, audit = sidecar.forward_step(
    frozen_hidden, return_diagnostics=True
)
# 初始化时 adapted 与 frozen_hidden 逐位相同。

limits = SidecarSafetyLimits(
    max_policy_kl=1e-3,
    max_output_linf_drift=0.05,
    max_state_linf_drift=0.1,
    max_transition_fro_drift=0.02,
)
```

`SidecarSafetyLimits` 只审计候选参数的局部变化。统一 Agent 可检查四项预算；
MiniCPM LoRA 桥只支持 policy KL 和输出漂移，若传入状态或 Cayley 转移预算会
显式报错。MiniCPM 的 `constancy_closure` 应返回固定旧域 prompt 上的原生
logits；这一 probe 才能检查冻结模型的行为守恒，酉性或双随机性本身不能代替它。

`carry_state=True` 适合 batch lane 与持续流一一对应的 token/序列；独立样本 minibatch 应设 `carry_state=False`，避免把不同样本隐状态串联。显式上下文在容量内获得稳定环分配；容量用尽默认报错，不会静默覆盖旧环。

内容/地址分离的路由记忆可单独使用：

```python
import torch
from mqr import ResidualAddressRouter, RoutedSlotMemory

router = ResidualAddressRouter(
    8, route_family="local_permutation", epsilon=0.75
)
memory = RoutedSlotMemory(8, 2, router=router)
state = memory.zero_state(
    1, device=torch.device("cpu"), dtype=torch.float32
)
prediction, state = memory.forward_step(
    state, write_value=torch.tensor([[1.0, -1.0]])
)
# prediction 来自写入前状态；只有地址推进，未寻址内容不会被平均。
```

运行无网络依赖的原理验证：

```bash
python3 test_online_learning.py
python3 test_routed_memory.py
python3 experiments/online_digits_principle.py
```

3 个种子上，共享单环的 A 遗忘为 `17.96±2.92` 个百分点，OGD-64 为 `14.63±3.32`，显式隔离双环为 `0.00±0.00`。OGD 同时降低了 B 的适应，双环参数翻倍；详细限制、数学证明和 2B 路线见 [`analysis/online_learning_research_plan.md`](analysis/online_learning_research_plan.md)。

### 时间型状态记忆

固定点 MQR 适合求稳定平衡，但充分松弛会擦除跨观察状态。时间型 MQR 每个
观察只执行一次更新，并把写入与遗忘解耦：

```python
import torch
from mqr import OnlineTemporalMQRClassifier

learner = OnlineTemporalMQRClassifier(
    input_dim=66, ring_dim=12, output_dim=10,
    core_kwargs={
        "leak_rates": (0.5, 0.1, 0.02, 0.005),
        "write_scales": (1.0, 1.0, 1.0, 1.0),
        "injection_rank": 24,
        "learn_transitions": False,
    },
    gate_input_dim=1,
    gate_kwargs={"rank": 8, "minimum_gate": 0.02},
    gate_lr=0.05, gate_positive_weight=8.25,
    lr=0.15, carry_state=True,
)
result = learner.online_step(
    torch.randn(1, 66), torch.tensor([3]),
    gate_input=torch.ones(1, 1), gate_target=torch.ones(1),
)
# result["logits"] 属于 theta_t；权重在返回前提交为 theta_(t+1)
# gate_target 不可能改变本步 gate/state/logits，只训练下一版本的门。
```

```bash
python3 test_temporal_mqr.py
python3 analysis/temporal_mqr_result_verify.py
python3 experiments/temporal_mqr_delayed_digits.py
```

3-seed 延迟 Digits 中，固定点 `K=24` 等价控制、单慢环和多尺度
unistochastic 的 held-out 准确率分别为 `9.06±1.43%`、`81.04±1.00%`
和 `77.50±1.13%`。将 `kappa=lambda` 改为独立强写入带来
`+55.31±7.10 pp` 的配对改善；但 unistochastic 比逐位同初始化的 identity
低 `1.25±0.83 pp`，尚无 Cayley 拓扑优势。完整协议和限制见
[`analysis/temporal_mqr_validation_report.md`](analysis/temporal_mqr_validation_report.md)。

### 实用在线事务与真实干扰

`write_gate` 可以是 `[batch]` 或 `[batch, num_timescales]`；它只控制新信息
写入，不改变旧状态的收缩率。交错会话使用显式 `stream_id`，状态库容量耗尽
默认报错；可选 LRU 会在返回值中报告确切淘汰会话。推理和反馈不必同步：

```python
prediction = learner.infer_step(
    torch.randn(1, 66),
    stream_id="session-42",
    write_gate=torch.tensor([1.0]),
)
# 先把 prediction["logits"] 返回给环境；标签稍后到达。
feedback = learner.apply_feedback(prediction["ticket_id"], torch.tensor([3]))
```

ticket 有容量、TTL、参数版本和单次消费语义，只精确重建签发时的**读出**梯度；
延迟更新完整 MQR/LoRA 仍需要参数快照或重算。21-run、3-seed 干扰 Digits
中，多尺度 oracle cue-only 门为 `73.44±5.16%`，无门为
`20.10±1.83%`，配对改善 `+53.33±6.17 pp`；同稀疏度随机门只有
`18.13±1.13%`。oracle 门由协议提供，尚未学得；多尺度仍未超过单慢环。

v4 进一步增加低秩辅助事件门：当前帧先写入和预测，随后才用 cue/non-cue
标签训练门；任务梯度在门处截断。新的 24-run、3-seed 实验中，44 参数学习门
为 `70.83±6.55%`，无门为 `19.86±1.68%`，不更新门为
`21.53±0.87%`；学习门相对无门配对提升 `+50.97±6.74 pp`，与 oracle
只差 `1.94±1.58 pp`。控制器只读取协议显式提供的非类别 marker，并收到即时
辅助标签，所以这不是无监督重要性发现或延迟奖励学习。单慢环仍高
`2.78±1.20 pp`。

```bash
python3 experiments/temporal_mqr_interference_digits.py
python3 analysis/temporal_mqr_interference_verify.py
python3 experiments/temporal_mqr_learned_gate_digits.py
python3 analysis/temporal_mqr_learned_gate_verify.py
```

严格事务、扰动上界、资源公式和结论边界见
[`analysis/temporal_mqr_runtime_report.md`](analysis/temporal_mqr_runtime_report.md)。

### 未来行为收益门（v5）

`UtilityDrivenMQR` 不再接收 cue/non-cue 标签。快环始终记录当前候选；慢环动作
只依赖当前新颖性、不确定性、过去预测误差、慢环饱和度和上下文变化。未来任务
标签到达后，成对影子轨迹给出
`CE(no_write) - CE(write) - memory_cost`，该数值通过签发时特征回放训练门：

```python
import torch
from mqr import UtilityDrivenMQR

learner = UtilityDrivenMQR(
    input_dim=64, ring_dim=24, output_dim=10,
    core_kwargs={"leak_rates": (1.0, 0.02),
                 "learn_transitions": False},
    utility_ogd_max_rank=8,
)
candidate = learner.observe(torch.randn(1, 64))
query = learner.observe(torch.zeros(1, 64), issue_candidate=False)
feedback = learner.resolve_utility(
    candidate["candidate_id"], torch.tensor([3])
)
# 正收益漏写会排队，并作为外部 forcing 在下一次 observe 时晋升慢环。
```

3-seed、随机目标位置、无 marker Digits 中，49 参数 OGD 收益门 held-out 为
`72.00±6.24%`，普通收益门为 `57.67±27.23%`，不更新门为
`10.33±3.79%`，全写为 `8.67±3.79%`，随机稀疏写为 `17.00±3.00%`；OGD
门的目标写入率/干扰误写率为 `100%/0%`，与因果 novelty 阈值和位置 oracle
相同。普通门在一个 seed 出现误写塌缩，所以 OGD 的 held-out 增益尚不稳定。
这个任务刻意让“唯一目标”可由新颖性
识别，因此证明的是从延迟任务收益学习一个**可观测规律**，不是预知任意未来
相关性。OGD 门逐票据正收益 AUPRC 仍只有 `75.19±2.89%`，且尚无等资源 GRU、
fast-weight、LoRA/replay 优势证据。

```bash
python3 test_utility_mqr.py
python3 experiments/temporal_mqr_utility_digits.py
python3 analysis/temporal_mqr_utility_verify.py
```

严格条件期望证明、不可能性下界、影子轨迹资源成本和 2B 近似路线见
[`analysis/utility_mqr_research_report.md`](analysis/utility_mqr_research_report.md)。

### 无 marker 反转、跨任务遗忘与基线（v6）

v6 增加只由过去输入更新的 `context_volatility` 迹，并在 rank-8
总门预算下硬路由两个 rank-4 收益专家。它不读取任务 ID、当前标签或
未来信息；单样本硬路由下，未选专家在普通更新和分块支撑 OGD 中都精确
零更新。五个未调参 seed 的 A→B→A 正式流比较了 GRU、fast-weight、
LoRA、OGD-LoRA 和等字节 replay。

在可塑参数 288–320、在线张量状态不超过 4096 B 的同上限下，双专家
MQR+OGD 的 A-after-B 下降为 `0.0 pp`，LoRA 为 `36.75 pp`；遗忘减少的
bootstrap 95% CI 为 `[34.00,39.50] pp`。其最终 A/B 平衡准确率高
`11.63 pp`，但全过程平均阶段准确率低 `8.58 pp`，且延迟约
`36.31 ms` 对 `0.48 ms`。MQR 总 sidecar 参数为 1354，LoRA 为 300；
因此证据支持独立的**稳定性机制**，不支持总体竞争力已成立。

```bash
python3 analysis/temporal_mqr_reversal_verify.py
```

完整预注册门槛、资源表和限制见
[`analysis/mqr_reversal_go_v2_report.md`](analysis/mqr_reversal_go_v2_report.md)。

### 统一四头 Agent 与严格 Go 核心对照

`TemporalUtilityMQRAgent` 将 placement、legality、pass、value 分头建模；策略只
使用学习到的合法性软先验，不应用规则 mask。离线阶段先做优势加权模仿，再以
1/4 学习率运行短轨迹 actor-critic。write critic 从严格后验的 matched
write/no-write 未来轨迹学习条件期望，不接收事件 marker。固定点 MQR 另提供
`implicit_update_from_output_gradient`，可接收任意已约简 `grad_logits`。

原 5-seed、3×3 实验使用输入相同而历史不同的超级劫配对。它验证了机制和失败门，
但棋盘过小，只保留为历史 smoke，不再作为围棋能力证据。

当前联合策略严格分解为
\(P(i)=P(\neg\mathrm{pass})P(i\mid\neg\mathrm{pass})\) 与
\(P(\mathrm{pass})=\sigma(s)\)。修正前把二元 pass score 与条件 placement logits
直接拼接，导致旧 10×10 文件的 pass 率和合法率失真；该文件已从证据链排除。

修正后的 10×10、5-seed 缩短 pilot 使用无损 306 维输入。MQR/identity/GRU/
fast-weight 的动态有效合法率为 `0.0153/0.0277/0.0552/0.0215`，平均回报为
`-17.125/-16.825/-16.425/-16.975`，历史焦点全部为 `0.500`。task-B 历史
probe 的 placement 教师一致率全部为 0；动态对局中 GRU/fast-weight 仍有约
`0.0046/0.0016` 的极低一致率。参数比 `1.0359`、估算 MAC 比 `1.0307` 通过门槛，但状态比
`5.333` 失败。因此 `mqr_effective=false`，这不是 MQR 能力增幅证据。

本地 MiniCPM5-1B 慢 LoRA 烟雾测试触发 1/3 次日程更新，适配器最大漂移
`3.71e-7`、冻结参数抽样漂移为 0；probe loss 反而变差，故只证明梯度链路。

```bash
python3 test_unified_mqr_agent.py
python3 experiments/unified_temporal_mqr_go.py
python3 analysis/unified_temporal_mqr_go_verify.py
python3 analysis/unified_temporal_mqr_minicpm_go_verify.py
python3 analysis/mqr_solver_certification.py
python3 analysis/mqr_sinkhorn_fair_compare.py
python3 analysis/mqr_sidecar_safety_verify.py
python3 experiments/frozen_sidecar_application_qualification.py
python3 analysis/mqr_application_positioning_verify.py
python3 analysis/mqr_mechanism_qualification_verify.py \
  analysis/results/mqr_mechanism_qualification_formal_10seed.json
python3 analysis/mqr_capacity_sweep_verify.py \
  analysis/results/mqr_cyclic_capacity_sweep_formal_10seed.json
```

10×10 历史 pilot 的复现命令、资源表和失败原因见
[`analysis/mqr_10x10_sidecar_pilot_report.md`](analysis/mqr_10x10_sidecar_pilot_report.md)。
其后完成的正式实验在 10×10 与 13×13 上各用 10 seeds，加入
identity、GRU、LSTM、fast-weight、LoRA、OGD-LoRA、replay-LoRA 和
Sinkhorn，并通过分配参数、状态字节、前向 MAC 和更新 MAC 的 1.05 门。
独立验证命令为：

```bash
python3 analysis/unified_temporal_mqr_go_competitive_verify.py \
  --require-replication \
  analysis/results/unified_temporal_mqr_go_competitive_10x10_formal_10seed.json \
  analysis/results/unified_temporal_mqr_go_competitive_13x13_formal_10seed.json
```

该实验证明了在线权重学习链路，但没有证明历史使用、规则获得或 MQR
独立优势，联合结论为 `mqr_effective=false`。权威表格和解释见
[`analysis/mqr_formal_competitive_go_report.md`](analysis/mqr_formal_competitive_go_report.md)；
整体数学与 sidecar 修改口径见
[`analysis/mqr_sidecar_revision_plan.md`](analysis/mqr_sidecar_revision_plan.md)。

### MiniCPM5 + Sayuri 围棋在线闭环

本地 `MiniCPM5-1B-AWQ-INT4` 作为冻结语义编码器：AWQ INT4 权重只在加载时解包为 FP16 执行张量，末层 `q_proj/v_proj` 挂接 43,008 个 FP32 LoRA 参数；外部两个 MQR 按黑/白上下文路由。Sayuri 保持为独立 GPLv3 GTP 进程，不把上游源码链接进 `mqr/`。

```bash
# 使用已经位于 UsedCode/Sayuri 的源码构建；不会再次 clone
bash scripts/setup_sayuri.sh
python3 test_go_minicpm.py

python3 experiments/minicpm_go_online.py \
  --teacher sayuri-policy --stream-mode game --rollout-policy teacher \
  --steps 24 \
  --metrics-path analysis/results/minicpm_go_sayuri_game_lora_24.json
```

每一步严格执行“先预测、后反馈更新”。24 步实测完成一局，held-out loss `3.289 → 3.170`，峰值 CUDA 显存约 `2.20 GB`，平均在线步约 `40–42 ms`；48 步基础局面流的回放 masked 教师一致率为 `75%`。相同 seed 的 no-LoRA 对照几乎完全一致，说明目前学习主要来自 MQR，尚无 LoRA 收益证据。checkpoint 会同步保存棋盘、环状态、路由与 OGD，已验证从 step 24 连续恢复。严格链式法则、分块 OGD 证明、完整指标和结论边界见 [`analysis/minicpm_sayuri_online_report.md`](analysis/minicpm_sayuri_online_report.md)。

多局因果实验使用 `preview_step` 先锁定旧权重动作，学生真正落子后才提交 Sayuri 反馈。原生 5×5 教师下，3 seed 的固定 probe loss 改善 `0.335±0.067`，但配对目差退化 `11.5±8.2`，原因是策略向 `pass` 塌缩。显式 pass 课程后对原生 Sayuri 的目差初步改善 `6.5±7.7`，但 raw 合法率和 probe 模仿指标未改善，小样本区间仍跨 0。正确规则也没有显著优于等结构错误规则。完整实验、不可识别性证明和分层策略方案见 [`analysis/minicpm_go_real_games_report.md`](analysis/minicpm_go_real_games_report.md)。

更严格的 v2 正式实验使用 5 个新 seed、4 个更新归因条件、每条件 8 局
训练和 8 局配对评估。外部合法性掩码后目差平均改善 `+5.75`，但
Student-t 95% CI 为 `[-2.25,+13.75]`。原生固定探针 loss 恶化 `0.209`、
pass Brier 恶化 `0.0087`，且学生回合未掩码合法率仅从 `8.96%` 到
`10.23%`。MQR+LoRA 与 MQR-only 的离散行为完全一致；LoRA-only 权重有
漂移但行为不变。因此当前行为变化归因于 MQR 动作头，仍不是内部规则
学习或稳定棋力提升证据。

```bash
python3 analysis/minicpm_go_real_games_v2_verify.py

python3 experiments/minicpm_go_real_games.py \
  --seeds 2026 2027 2028 \
  --conditions rules-online rules-no-learning board-only-online rules-mqr-only \
  --train-games 4 --eval-games 2 --probe-positions 16 \
  --metrics-path analysis/results/minicpm_go_real_games_3seed.json
```

### CIFAR-100训练

#### 基础训练 (使用AdamW优化器)

```bash
python train_mobius_cifar100.py \
    --embed-dim 384 \
    --depth 20 \
    --alpha 0.1 \
    --lora-rank 16 \
    --readout-dim 16 \
    --batch-size 64 \
    --epochs 200 \
    --lr 3e-4 \
    --mixup-alpha 0.2 \
    --mixup-prob 0.5 \
    --ortho-loss-weight 0.01
```

#### BPTT-free 精确隐式更新（保留 `--use-eqprop` 兼容名称）

该模式执行“推理固定点 + 伴随固定点 + 精确 Cayley 坐标拉回”。历史 HTML 中的酉更新遗漏了 `H=|U|²` 与 Cayley 微分的正确链式法则，当前实现已修正；严格推导与旧公式反例见 [`analysis/mqr_math_proof.md`](analysis/mqr_math_proof.md)。

```bash
python train_mobius_cifar100.py \
    --use-eqprop \
    --embed-dim 384 \
    --depth 20 \
    --alpha 0.1 \
    --lora-rank 16 \
    --readout-dim 16 \
    --eqprop-adjoint-steps 20 \
    --eqprop-unitary-lr-ratio 0.5 \
    --eqprop-injection-lr-ratio 1.0 \
    --eqprop-readout-lr-ratio 1.0 \
    --batch-size 64 \
    --epochs 200 \
    --lr 3e-4
```

#### Online 双环扩展（冻结世界模型 + 可学习策略 + 可学习目标态，可选原型读出）

```bash
python train_mobius_cifar100.py \
    --use-eqprop \
    --base-unitary-init random --base-unitary-seed 123 \
    --eqprop-learnable-state-targets \
    --eqprop-state-target-weight 0.1 \
    --readout-mode proto --proto-tau 1.0 \
    --embed-dim 384 --depth 20 --alpha 0.1 \
    --lora-rank 16 --readout-dim 16 \
    --batch-size 64 --epochs 200 --lr 3e-4
```

#### 表达能力扩展（按 A→B→C 顺序，可与 EQPROP/双环组合）

**A. Patch Embedding（推荐先开）**

```bash
python train_mobius_cifar100.py \
    --use-eqprop \
    --image-encoder patch --patch-size 4 --patch-embed-dim 256 --patch-pool mean \
    --embed-dim 384 --depth 20 --alpha 0.1 \
    --lora-rank 64 --readout-dim 32 \
    --batch-size 64 --epochs 200 --lr 3e-4
```

**B. 自保持混合 \(H_{eff}=(1-\beta)I+\beta H\)**（抑制“过度平均化”）

```bash
python train_mobius_cifar100.py \
    --use-eqprop \
    --image-encoder patch --patch-size 4 --patch-embed-dim 256 --patch-pool mean \
    --h-mix-beta 0.7 \
    --embed-dim 384 --depth 20 --alpha 0.1 \
    --lora-rank 64 --readout-dim 32 \
    --batch-size 64 --epochs 200 --lr 3e-4
```

**C. 复数酉推理（phase 参与 inference）**：`dynamics_mode=unitary` + `measurement=abs`

```bash
python train_mobius_cifar100.py \
    --use-eqprop \
    --image-encoder patch --patch-size 4 --patch-embed-dim 256 --patch-pool mean \
    --dynamics-mode unitary --measurement abs \
    --inj-activation relu \
    --embed-dim 384 --depth 20 --alpha 0.1 \
    --lora-rank 64 --readout-dim 32 \
    --batch-size 64 --epochs 200 --lr 3e-4
```

#### 使用哈密顿优化器训练

```bash
python train_mobius_cifar100.py \
    --use-hamiltonian \
    --embed-dim 384 \
    --depth 20 \
    --alpha 0.1 \
    --lora-rank 16 \
    --readout-dim 16 \
    --batch-size 64 \
    --epochs 200 \
    --lr 3e-4
```

#### 继续训练（Resume Training）

**方式1：指定checkpoint路径 + 额外epoch数**

```bash
# 从epoch 199继续训练100个epoch（到epoch 299）
python train_mobius_cifar100.py \
    --resume /tmp/mqr_ablation_opt_unistochastic_flatten_10ep/checkpoint_epoch_199.pth \
    --extra-epochs 100 \
    --save-dir /tmp/mqr_ablation_opt_unistochastic_flatten_10ep \
    [其他参数保持与之前训练相同]
```

**方式2：指定checkpoint路径 + 新的总epoch数**

```bash
# 从checkpoint继续训练到总共300 epochs
python train_mobius_cifar100.py \
    --resume /tmp/mqr_ablation_opt_unistochastic_flatten_10ep/checkpoint_epoch_199.pth \
    --epochs 300 \
    --save-dir /tmp/mqr_ablation_opt_unistochastic_flatten_10ep \
    [其他参数保持与之前训练相同]
```

**方式3：自动恢复（自动查找save-dir中最新的checkpoint）**

```bash
# 自动从save-dir中找到最新checkpoint并继续训练100个额外epoch
python train_mobius_cifar100.py \
    --auto-resume \
    --extra-epochs 100 \
    --save-dir /tmp/mqr_ablation_opt_unistochastic_flatten_10ep \
    [其他参数保持与之前训练相同]
```

> **注意**：续训练时，optimizer和scheduler状态会自动恢复，确保学习率等参数的连续性。

### 恢复训练

```bash
python train_mobius_cifar100.py \
    --resume ./checkpoints/mobius_quantum_ring_best.pth \
    --epochs 200
```

## 🔧 参数说明

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--embed-dim` | 384 | 环的维度 N（节点数/隐藏维度） |
| `--depth` | 20 | 固定点松弛迭代步数 K（legacy 名称；更深通常更稳但更慢） |
| `--relaxation-steps` | None | 覆盖 `--depth` 的别名（更贴近算法语义） |
| `--image-encoder` | flatten | 图像编码方式：`flatten`（像素展平，严格HTML默认）/ `patch`（卷积patch embedding）/ `vit`（成熟ViT backbone，推荐用于追求高精度） |
| `--patch-size` | 4 | `--image-encoder patch/vit` 时的 patch 尺寸（conv kernel=stride=patch\_size） |
| `--patch-embed-dim` | 256 | `patch` 编码输出通道数（patch token维度） |
| `--patch-pool` | mean | patch token 聚合：`mean`（推荐）或 `flatten`（更大输入维度） |
| `--vit-dim` | 384 | `vit` token维度（仅 `--image-encoder vit` 生效；建议与 `--embed-dim` 一致） |
| `--vit-depth` | 6 | ViT transformer blocks 数（仅 `--image-encoder vit` 生效） |
| `--vit-heads` | 6 | ViT 注意力头数（仅 `--image-encoder vit` 生效） |
| `--vit-mlp-dim` | 1536 | ViT MLP隐藏维度（仅 `--image-encoder vit` 生效） |
| `--vit-dropout` | 0.0 | ViT dropout（仅 `--image-encoder vit` 生效） |
| `--vit-pool` | cls | ViT pooling：`cls`（CLS token）或 `mean`（patch tokens 均值） |
| `--alpha` | 0.1 | 耗散/注入系数 \(\alpha\in(0,1]\) |
| `--lora-rank` | 16 | LoRA 注入 rank：\(\mathcal{J}(x)=W_{up}W_{down}x\) |
| `--inj-activation` | none | 注入端激活：\(\mathcal{J}(x)=W_{up}\,\sigma(W_{down}x)\)，可选 `none/relu/tanh/gelu` |
| `--state-activation` | none | 环内激活：\(h\leftarrow\sigma((1-\alpha)hH^T+\alpha\mathcal{J}(x))\)，可选 `none/relu/tanh`（unitary模式需为none） |
| `--h-mix-beta` | 1.0 | 自保持混合系数：\(H_{eff}=(1-\beta)I+\beta H\)（\(\beta=1\)为严格HTML默认） |
| `--learnable-h-mix-beta` | False | 令 \(\beta\) 可学习（eqprop模式下用手工梯度更新；autograd模式下由优化器更新） |
| `--h-mix-beta-lr-ratio` | 1.0 | \(\beta\) 学习率比例（eqprop模式） |
| `--dynamics-mode` | unistochastic | 推理动力学：`unistochastic`（默认 \(H=|U|^2\)）或 `unitary`（复数域 \(U^\dagger\) 推理） |
| `--measurement` | identity | 读出/状态损失的测量：`identity/abs/real`（unitary 下 identity 自动视为 abs） |
| `--readout-dim` | 16 | 局部采样大小 \(|\mathcal{S}|\)（默认采样前 k 个节点） |
| `--readout-mode` | linear | 读出模式：`linear`（线性局部读出）或 `proto`（原型距离 logits；需要 `--eqprop-learnable-state-targets`） |
| `--proto-tau` | 1.0 | 原型距离 logits 温度参数 \(\tau\)（仅 `--readout-mode proto` 生效） |
| `--num-heads` | 8 | **兼容参数（当前MQR实现中忽略）** |

### CIFAR-100 推荐默认超参（flatten-patch 作为推荐配置）

下面这组配置在 10 epoch 的快速消融中可将 Test Acc 提升到约 **10%+**（相比 mean-pool 版本显著更好），建议作为后续实验的起点：

| 模块 | 推荐值 | 说明 |
|------|--------|------|
| Image encoder | `--image-encoder patch --patch-size 4 --patch-embed-dim 128 --patch-pool flatten` | **保留空间信息**（不做 mean pool） |
| Ring size | `--embed-dim 384 --depth 12 --alpha 0.3` | 更强注入（更快收敛/更大有效信号） |
| Injection | `--lora-rank 256 --inj-activation relu` | 提升容量 + 注入非线性 |
| Relaxation nonlinearity | `--state-activation tanh` | 1-Lipschitz，收敛证明仍成立 |
| Over-mixing control | `--h-mix-beta 0.6 --learnable-h-mix-beta` | 抑制双随机“过度平均化” |
| Readout | `--readout-mode linear --readout-dim 128` | 先用最稳定的线性读出做基线 |
| EQProp LRs | `--lr 3e-3 --eqprop-unitary-lr-ratio 0.2 --eqprop-readout-lr-ratio 5.0 --eqprop-encoder-lr-ratio 5.0` | 让 encoder/readout 真正能学起来 |

### 目标 70%+（C策略）：ViT backbone + MQR head（逐步替代）

如果你的目标是“证明 MQR 训练/结构可以替代深网模块并达到可用精度”，推荐走 **C 策略**：用成熟的 ViT 提供层级表征，MQR 作为 head（或后续逐步扩张替代更大比例的模块）。

- **动机**：当前 `patch_embed + flatten` 仍然缺少“多层抽象特征学习”，导致上限偏低、收敛慢。
- **改动点**：新增 `--image-encoder vit`，并在 `--use-eqprop` 下用 `--eqprop-encoder-optim adamw` 训练 ViT，ring 仍使用严格 EQProp。
- **预期效果**：显著提升特征可分性与收敛速度，为后续“扩大 MQR 占比”提供可行基线。

示例命令（建议先跑 50~100 epoch 看上升趋势，再拉长到 300+）：

```bash
python train_mobius_cifar100.py --epochs 500 --batch-size 64 --num-workers 4 --use-eqprop --image-encoder vit --patch-size 4 --vit-dim 192 --vit-depth 4 --vit-heads 3 --vit-mlp-dim 768 --vit-dropout 0.0 --vit-pool cls --embed-dim 192 --depth 12 --alpha 0.3 --lora-rank 128 --inj-activation relu --state-activation tanh --h-mix-beta 0.6 --learnable-h-mix-beta --h-mix-beta-lr-ratio 1.0 --eqprop-encoder-lr-ratio 0.2 --eqprop-encoder-optim adamw --readout-dim 64 --readout-mode linear --lr 3e-3 --warmup-epochs 1 --min-lr-ratio 0.01 --label-smoothing 0.1 --cutmix-prob 0.5 --cutmix-alpha 1.0 --mixup-prob 0.5 --mixup-alpha 0.2 --eqprop-adjoint-steps 10 --eqprop-unitary-lr-ratio 0.2 --eqprop-injection-lr-ratio 1.0 --eqprop-readout-lr-ratio 5.0 --base-unitary-init identity --save-dir ./checkpoints/mqr_vit_eqprop_recipe_smoke --save-freq 50 --auto-resume
```

### BPTT-free 隐式更新参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use-eqprop` | False | 启用固定点/伴随隐式更新（历史兼容名称，不使用 BPTT） |
| `--eqprop-adjoint-steps` | 20 | 伴随态 \(h^\dagger\) 固定点求解迭代步数 |
| `--eqprop-unitary-lr-ratio` | 0.5 | 酉流形参数（A_real/A_imag）的学习率比例 |
| `--eqprop-injection-lr-ratio` | 1.0 | LoRA 注入参数学习率比例 |
| `--eqprop-readout-lr-ratio` | 1.0 | 读出层参数学习率比例 |
| `--eqprop-learnable-state-targets` | False | 启用可学习目标平衡态（按类原型 \(P\in\mathbb{R}^{C\times N}\)） |
| `--eqprop-state-target-weight` | 0.0 | 目标态损失权重（\(\LL_{state}=\frac12\|h^\*-p_y\|^2\)），需要 `--eqprop-learnable-state-targets` |
| `--eqprop-state-target-lr-ratio` | 1.0 | 目标态参数学习率比例 |
| `--base-unitary-init` | identity | 冻结世界模型酉矩阵初始化：`identity`（兼容默认）或 `random` |
| `--base-unitary-scale` | 0.01 | `random` 初始化时 skew-Hermitian 采样尺度（Cayley） |
| `--base-unitary-seed` | None | `random` 初始化随机种子（可复现） |
| `--eqprop-encoder-lr-ratio` | 1.0 | EQProp 模式下 encoder 参数（如 patch embedding）的学习率比例 |
| `--eqprop-encoder-optim` | adamw | EQProp 模式下 encoder 的优化器：`adamw/sgd/none`（ring 仍严格EQProp） |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 200 | 训练轮数 |
| `--batch-size` | 64 | 批次大小 |
| `--lr` | 3e-4 | 学习率 |
| `--warmup-epochs` | 5 | warmup 轮数（warmup→cosine） |
| `--min-lr-ratio` | 0.01 | cosine 最小学习率比例（eta\_min = lr * min\_lr\_ratio） |
| `--weight-decay` | 0.05 | 权重衰减 |
| `--unitary-lr-ratio` | 0.5 | 酉矩阵参数学习率比例 |
| `--label-smoothing` | 0.0 | 标签平滑系数（eqprop模式通过 soft target 实现） |

### 数据增强（DeiT/ViT常用）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mixup-prob` | 0.5 | Mixup 概率 |
| `--mixup-alpha` | 0.2 | Mixup alpha |
| `--cutmix-prob` | 0.0 | CutMix 概率（先于 mixup 采样） |
| `--cutmix-alpha` | 1.0 | CutMix alpha |

### 哈密顿优化

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use-hamiltonian` | False | 是否使用哈密顿优化器 |

### Mixup数据增强

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mixup-alpha` | 0.2 | Mixup Beta分布参数 |
| `--mixup-prob` | 0.5 | Mixup应用概率 |

### 正交约束

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ortho-loss-weight` | 0.01 | 诊断用“酉性误差”权重（`||U^†U - I||_F`，通常可设为0） |

## 📊 监控训练

使用TensorBoard监控训练过程:

```bash
tensorboard --logdir runs/
```

监控指标包括:
- `Loss/Train`: 训练损失
- `Loss/Test`: 测试损失
- `Loss/Ortho`: 正交约束损失
- `Accuracy/Train`: 训练准确率
- `Accuracy/Test`: 测试准确率
- `Accuracy/Test_ClassAvg`: 平均类别准确率

## 🎯 核心模块 (Core Modules)

- `mqr/unitary.py`：`CayleyUnistochasticParam`（构造 \(U\) 与 \(H=|U|^2\)）
- `mqr/ring.py`：`MoebiusQuantumRing`（固定点松弛、LoRA 注入、局部采样读出）
- `mqr/baselines.py`：`SinkhornDoublyStochasticParam`（log-space 前向与收敛隐式拉回）
- `mqr/online.py`：`OnlineMultiRingClassifier` 与 `OrthogonalGradientMemory`
- `mqr/temporal.py`：单步快慢环、零扰动 residual sidecar、有符号状态证书
- `mqr/routing.py`：内容槽/地址分离、残差稀疏路由、硬写预算与校准指标
- `mqr/safety.py`：候选更新预算、策略 KL、输出漂移与 residual 诊断
- `mqr/utility.py`：未来收益回归、成对影子状态和延迟候选晋升
- `mqr/agent.py`：四头在线事务、慢 Cayley 时标、constancy 回滚和 LoRA 巩固
- `mobius_quantum_ring.py`：向后兼容门面（旧脚本无需改 import）

## 📈 已验证性质与结论边界

当前可严格支持的结构优势：

1. **固定点稳定性**：阻尼双随机/酉传播构成压缩映射
2. **约束保持**：Cayley 参数化始终生成酉矩阵，`H=|U|²` 始终双随机
3. **低反向内存**：伴随固定点无需保存全部松弛轨迹
4. **参数效率**：低秩注入适合作为冻结主干的 sidecar；是否优于 LoRA 仍需等参数实验
5. **坐标覆盖**：最小 Cayley 坐标完整覆盖 \(\mathfrak u(N)\)，但
   \(U\mapsto|U|^2\) 最多只有 \((N-1)^2\) 个局部可见方向，identity 点一阶秩为 0
6. **安全事务**：有符号时间状态的逐步 \(\ell_\infty\) 上界、Cayley
   有限漂移界和候选更新原子回滚均已数值认证；它们是安全性质，
   不是任务性能优势

当前不支持的结论：MQR 优于 GRU/fast-weight/LoRA/replay、已学会围棋规则、可提升
1B/2B 语言生成能力，或具备一般动物学习能力。

当前最合理的应用定位是冻结小模型的有界 residual sidecar：保存会话状态，接收
预测后的反馈，只在旧域 KL、状态、输出和转移预算内提交个性化或纠错更新。10-seed
合成资格实验中，有状态冲突偏好准确率为 `1.000`，清空状态为 `0.500`，受保护
旧域 KL 最大 `2.86e-16`；这仍只是应用机制证据。协议与正式竞争力缺口见
[`analysis/mqr_application_positioning_report.md`](analysis/mqr_application_positioning_report.md)。

最新等资源正式实验已经补齐此前缺口：10×10 与 13×13 各使用 10 个预注册
seeds、每任务 128 次主在线反馈，并加入 identity、GRU、LSTM、fast-weight、
LoRA、OGD-LoRA、replay-LoRA 与隐式 Sinkhorn。所有方法的循环状态均为精确
48 B；参数、总前向 MAC、分配更新 MAC 的最大比分别不超过
`1.0431/1.0304/1.0130`（10×10）和 `1.0398/1.0253/1.0127`
（13×13）。MQR 的 held-out loss 在两尺度都显著下降，证明在线更新链有效；但
历史焦点准确率仍为 `0.500`，MQR 与 identity/多个时序核心基本同轨，13×13
还在动态合法率和回报上显著落后 replay-LoRA。因此正式联合结论仍是
`mqr_effective=false`。完整协议、表格和停止门见
[`analysis/mqr_formal_competitive_go_report.md`](analysis/mqr_formal_competitive_go_report.md)。

为判断负结果是否只由 Go 协议或 48 B 容量造成，又完成了 10-seed、
长度 12、三次重复 query 的 schema-v2 机制资格实验，以及 48/192/768 B
容量扫描。未来损失的梯度确实达到注入和环转移，但 cyclic MQR
历史准确率 `0.5375` 显著低于 identity 的 `0.5875`；三档等状态扫描也
都没有击败 identity，768 B 时配对差为
`-0.07125 [-0.13132,-0.01118]`。摊销资源门通过，但三档单次
Cayley 刷新峰值比 `1.1596/1.7122/2.3110` 全部失败。该结果排除了
“只是状态太小”的解释。详见
[`analysis/mqr_mechanism_and_capacity_report.md`](analysis/mqr_mechanism_and_capacity_report.md)。

针对该失败，新版将 MQR 从“混合内容”改为“路由地址”。10-seed、64-episode/
seed 的无显式 marker 长延迟实验中，残差路由 MQR 的正向、上下文反转和重复查询
准确率均为 `1.000`；相对 identity、GRU、fast-weight 的配对总体准确率差分别为
`0.43379 [0.42028,0.44729]`、`0.49248 [0.48207,0.50289]` 和
`0.35176 [0.32264,0.38087]`。identity replacement 的降幅为
`0.43379 [0.42028,0.44729]`，event/distractor 写率为 `1.000/0.000`，五轴
MQR/control 资源比均不超过 `1.05`。因此“路由机制资格”通过。

但直接固定环形缓冲区也得到 `1.000`，配对差严格为 0；当前事件/干扰幅值还容易
区分，MQR 路由本身也使用已知循环置换。这意味着独立算法优势仍为 false。完整
证明、成本、限制和下一阶段随机拓扑实验见
[`analysis/mqr_routed_memory_report.md`](analysis/mqr_routed_memory_report.md)。

Phase IV 已实际执行上述随机拓扑门。每个 seed 生成两个不透明、不同的局部交换
拓扑；事件与干扰在每个样本内是相同 `(key,value)` 多重集的精确重排，写门只用
过去 cue 与当前候选的关系，并以真实未来 query loss 的 write/no-write 差训练。
10 seeds 中，稀疏 Givens MQR 与分块 Cayley 的最终准确率和拓扑恢复率均为
`1.000`；相对 identity 和直接 ring buffer 的配对准确率优势分别为
`0.28730 [0.27920,0.29541]` 与 `0.49707 [0.49112,0.50302]`，identity
replacement 显著伤害性能，A→B→A 的 A 参数漂移严格为 0。

然而，同图、同 8 个路由参数、同 32 B 状态和同解析计算预算的可学习稀疏置换
也为 `1.000`，MQR 配对差严格为 0。进一步把强基线学习率从 0.1 调至 0.2 后，
其 B 阶段 prequential regret 从 `0.10564` 降至 `0.07230`，反而低于 Givens 的
`0.08826`；这与局部关系 `p=sin²(theta)=sigmoid(z)` 的完全表达等价一致。
所以本轮只通过 `contextual_route_mechanism_qualified`，不通过
`mqr_independent_algorithm_advantage`。权威解释和复现命令见
[`analysis/mqr_contextual_topology_report.md`](analysis/mqr_contextual_topology_report.md)。

## 🔬 实验对比

### 与 Sinkhorn/Birkhoff 构造的结构对照

下表只比较约束构造，不代表端到端性能或速度结论。

| 特性 | mHC | Möbius Ring |
|------|-----|-------------|
| 连接集合 | Birkhoff多面体 | unistochastic 子集 \(H=|U|^2\) |
| 约束构造 | Sinkhorn迭代 | Cayley 稠密求解 + 模平方 |
| 计算复杂度 | O(kn²) | O(n³)（稠密线性求解，可缓存） |
| 相位信息 | 无 | 标准 \(H\) 路径丢弃；仅复数 \(U\) 动力学保留 |
| 量子效应 | 无 | 无（经典 unistochastic 参数化） |

### 与传统Transformer对比

| 特性 | ViT | Möbius Ring |
|------|-----|-------------|
| 权重约束 | 无 | 双随机 + 酉性 |
| 优化方法 | SGD/Adam | Autograd 或固定点隐式梯度 |
| 特征融合 | 加法 | 阻尼固定点混合 |
| 数值稳定性 | 依实现而定 | 环内部收敛有条件保证 |

## 📖 参考文献

本实现基于HTML文档"Möbius Quantum Ring.html"中的理论设计:

1. **流形与李代数**
   - 流形: 局部欧几里得空间
   - 李代数 𝔬(3): 反对称矩阵集合
   - 酉群 U(n): U^† U = I

2. **DeepSeek mHC论文**
   - mHC: Manifold-Constrained Hyper-Connections
   - Birkhoff多面体: 双随机矩阵集合
   - Sinkhorn-Knopp算法

3. **本方案改进**
   - 使用酉矩阵替代双随机矩阵
   - H = |U|² 自动满足双随机性质
   - 以酉参数化生成 unistochastic 连接；“量子”仅为历史命名

## 🤝 贡献

欢迎提出问题和改进建议!

## 📄 许可证

MIT License

## 🙏 致谢

- HTML文档作者提供了详细的理论设计
- DeepSeek团队mHC论文的启发
- PyTorch团队提供的深度学习框架
