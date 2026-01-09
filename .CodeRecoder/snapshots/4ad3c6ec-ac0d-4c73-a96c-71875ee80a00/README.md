# 莫比乌斯量子环形网络 (Möbius Quantum Ring Network)

基于HTML文档中 **MQR / UHR-Net (Möbius Quantum Ring / Unistochastic Hamiltonian Ring)** 架构的PyTorch实现，严格复现其核心动力学：**Cayley 酉矩阵参数化 → 幺模随机连接 \(H=|U|^2\) → 固定点松弛推理 → LoRA式注入 → 局部采样读出**。

## 🌟 核心创新

### 网络架构 (Architecture)

- **幺模随机流形约束 (Unistochastic Manifold)**：学习一个酉矩阵 \(U\in U(N)\)，并用 \(H=|U|^2\) 作为连接矩阵（天然双随机、能量守恒）。
- **无头无尾环形动力学 (Headless Ring Dynamics)**：推理不是“前馈计算图”，而是通过固定点松弛迭代收敛到唯一稳态 \(h^\*\)。
- **哈密顿注入/LoRA 注入 (Hamiltonian / LoRA Injection)**：输入通过低秩注入算子驱动环形系统：\(\mathcal{J}(x)=W_{up}W_{down}x\)。
- **局部投影采样读出 (Local Projective Sampling)**：只采样环上局部节点集合 \(\mathcal{S}\)（默认前 k 个节点）得到输出：\(y=W_{readout}\cdot h^\*_{\mathcal{S}}\)。
- **双酉矩阵解耦 (Dual-Unitary World/Policy, optional)**：将连接酉矩阵分解为 \(U_{total}=U_{policy}\,U_{base}\)，其中 \(U_{base}\) 冻结作为稳定“世界模型”，\(U_{policy}\) 可学习作为“策略”。
- **可学习GT平衡态 (Learnable Goal Equilibrium, optional)**：引入按类/按任务的可学习目标平衡态 \(p_y\)，用于状态级目标对齐（模拟“受教育/反思”过程）。
- **原型距离读出 (Prototype Readout, optional)**：可选用 \(\hat{y}_{c}=-\|h^\*_{\mathcal{S}}-p_{c,\mathcal{S}}\|^2/(2\tau)\) 直接产生 logits，将分类目标与目标平衡态绑定。

### 算法改进 (What changed in this repo)

- **从“类ViT堆叠结构”重构为 HTML 中的 MQR 固定点环**：当前工程的实现以 `Möbius Quantum Ring.html` 为唯一算法规范，核心模块已迁移到 `mqr/` 包中。

### 1. **酉矩阵参数化 (Unitary Matrix Parameterization)**
- 使用 Cayley 变换从 **反埃尔米特矩阵 (skew-Hermitian)** 生成酉矩阵：`U = (I - A)(I + A)^(-1)`
- 保证 `U^† U = I`
- 避免 Sinkhorn 迭代（mHC 风格）的归一化开销

### 2. **双随机权重自动生成**
- 根据量子力学原理,酉矩阵的模平方自动构成双随机矩阵: `H = |U|²`
- 自动满足:
  - `H_ij ≥ 0` (非负性)
  - `∑_j H_ij = 1` (行和为1)
  - `∑_i H_ij = 1` (列和为1)
- 保证了信号传播的能量守恒,防止梯度爆炸/消失

### 3. **推理环：固定点松弛 (Inference = Relaxation to Fixed Point)**
- 推理环(实数域)通过迭代收敛到稳态：
  - \(h \leftarrow (1-\alpha)\,h\,H^T + \alpha\,\mathcal{J}(x)\)
- \(\alpha\in(0,1]\) 是耗散/注入平衡系数，保证收敛稳定（Banach 不动点思想）。

### 4. **更新环：伴随态/相位更新（实现为工具函数）**
- HTML 中给出了“伴随均衡态 \(h^\dagger\)”与 \(\partial\mathcal{L}/\partial H\approx h^\dagger\otimes h^\*\) 的推导。
- 工程中提供了对应的诊断工具（默认训练仍使用 PyTorch autograd）：
  - `MoebiusQuantumRing.compute_adjoint_state(...)`
  - `MoebiusQuantumRing.approx_grad_H(...)`

## 📁 项目结构

```
MöbiusQuantumRing/
├── mqr/                          # 工程化核心实现（按HTML算法复现）
│   ├── unitary.py                 # CayleyUnistochasticParam (U, H=|U|^2)
│   ├── ring.py                    # MoebiusQuantumRing / injection / readout
│   └── __init__.py
├── mobius_quantum_ring.py         # 向后兼容门面（re-export mqr/）
├── train_mobius_cifar100.py       # CIFAR-100训练脚本
├── quick_start.py                 # 快速开始示例
├── test_mobius_model.py           # 测试套件
├── README.md                       # 本文档
└── Möbius Quantum Ring.html       # 架构设计文档
```

## 🚀 快速开始

### 安装依赖

```bash
pip install torch torchvision numpy tensorboard
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

#### 严格复现 HTML：推理/训练同步（Holomorphic Equilibrium Propagation, **不使用BPTT**）

该模式严格按 `Möbius Quantum Ring.html` 的“推理固定点 + 伴随态固定点 + 李代数更新(ΔA)”执行更新：

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
| `--depth` | 12 | 固定点松弛迭代步数 K（legacy 名称） |
| `--relaxation-steps` | None | 覆盖 `--depth` 的别名（更贴近算法语义） |
| `--alpha` | 0.1 | 耗散/注入系数 \(\alpha\in(0,1]\) |
| `--lora-rank` | 16 | LoRA 注入 rank：\(\mathcal{J}(x)=W_{up}W_{down}x\) |
| `--readout-dim` | 16 | 局部采样大小 \(|\mathcal{S}|\)（默认采样前 k 个节点） |
| `--readout-mode` | linear | 读出模式：`linear`（线性局部读出）或 `proto`（原型距离 logits；需要 `--eqprop-learnable-state-targets`） |
| `--proto-tau` | 1.0 | 原型距离 logits 温度参数 \(\tau\)（仅 `--readout-mode proto` 生效） |
| `--num-heads` | 8 | **兼容参数（当前MQR实现中忽略）** |

### EQPROP（严格HTML）参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use-eqprop` | False | 启用严格“推理/训练同步”模式（不使用BPTT） |
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

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 200 | 训练轮数 |
| `--batch-size` | 64 | 批次大小 |
| `--lr` | 3e-4 | 学习率 |
| `--weight-decay` | 0.05 | 权重衰减 |
| `--unitary-lr-ratio` | 0.5 | 酉矩阵参数学习率比例 |

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
- `mobius_quantum_ring.py`：向后兼容门面（旧脚本无需改 import）

## 📈 预期性能

基于HTML文档中的理论分析,该架构相比传统方法的优势:

1. **训练稳定性**: 双随机权重保证信号能量守恒
2. **收敛速度**: 酉矩阵约束减少搜索空间
3. **泛化能力**: 相位干涉提供更丰富的特征表达
4. **量子隧穿效应**: 相位旋转允许跳出局部最优

## 🔬 实验对比

### 与mHC (Manifold-Constrained Hyper-Connections)对比

| 特性 | mHC | Möbius Ring |
|------|-----|-------------|
| 流形约束 | Birkhoff多面体 | 酉群流形 |
| 投影算法 | Sinkhorn迭代 | Cayley变换(解析式) |
| 计算复杂度 | O(kn²) | O(n³) (一次SVD) |
| 相位信息 | 无 | 保留 |
| 量子效应 | 无 | 有(相位干涉) |

### 与传统Transformer对比

| 特性 | ViT | Möbius Ring |
|------|-----|-------------|
| 权重约束 | 无 | 双随机 + 酉性 |
| 优化方法 | SGD/Adam | 哈密顿动力学 |
| 特征融合 | 加法 | 相位干涉 |
| 数值稳定性 | 可能不稳定 | 理论保证 |

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
   - 相位干涉实现量子隧穿效应

## 🤝 贡献

欢迎提出问题和改进建议!

## 📄 许可证

MIT License

## 🙏 致谢

- HTML文档作者提供了详细的理论设计
- DeepSeek团队mHC论文的启发
- PyTorch团队提供的深度学习框架
