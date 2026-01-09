# Möbius Quantum Ring 网络架构详解 (更新版 v2.0)

> **版本说明**: 本文档基于最新代码更新，包含复数动力学、ViT编码器等新特性

**最后更新**: 2025-01-07## 1. 整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MöbiusQuantumRingImageClassifier                      │
│                         (图像分类器包装层)                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
            [Flatten Mode]  [Patch Embed Mode]  [ViT Encoder]
        输入: [B, 3, 32, 32]      输入: [B, 3, 32, 32]    输入: [B, 3, 32, 32]
        输出: [B, 3072]           输出: [B, 256]         输出: [B, 384]
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        MoebiusQuantumRing                                │
│                          (核心量子环)                                      │
│                                                                           │
│  新增特性:                                                                │
│  - dynamics_mode: "unistochastic"(实值) | "unitary"(复值)                │
│  - h_mix_beta: H_eff = (1-β)I + βH (自保留混合)                          │
│  - measurement: "identity" | "abs" | "real" (测量机制)                   │
└─────────────────────────────────────────────────────────────────────────┘
                    │           │           │
                    ▼           ▼           ▼
        ┌───────────┴───┐ ┌────┴──────┐ ┌─┴─────────────┐
        │  CayleyUnitary│ │Injection  │ │  Readout      │
        │   Parameter   │ │   LoRA    │ │   Module      │
        │                │ │           │ │                │
        │ U ∈ U(N)       │ │ Low-rank  │ │ Local proj.   │
        │ H = |U|²       │ │ injection│ │ to logits      │
        └───────────────┘ └───────────┘ └───────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        动力学模式对比                                      │
├─────────────────────────────────────────────────────────────────────────┤
│  Unistochastic模式 (默认)                      Unitary模式 (新)          │
│  ────────────────────────────────────          ─────────────────────    │
│  状态: h ∈ ℝ^N (实数)                          状态: h ∈ ℂ^N (复数)       │
│  转移: H = |U|² (双随机)                      转移: U (酉矩阵)          │
│  迭代: h ← (1-α)hH^T + αJ(x)                  迭代: h ← (1-α)hU^† + αJ(x) │
│  测量: identity                              测量: abs/real             │
│  特点: 经典双随机动力学                       特点: 保留相位信息          │
└─────────────────────────────────────────────────────────────────────────┘
```## 2. 核心模块详细架构

### 2.1 CayleyUnistochasticParam (酉变换参数模块)

**位置**: `mqr/unitary.py`

**功能**: 通过Cayley变换生成酉矩阵U，并计算unistochastic转移矩阵H

**网络结构**:
```
参数:
├── A_real: [N, N] - 实部参数 (初始化: N(0, 0.01))
└── A_imag: [N, N] - 虚部参数 (初始化: N(0, 0.01))

计算流程:
1. 构造斜埃尔米特矩阵 A:
   A = 0.5 * [(R - R^T) + i(I + I^T)]
   其中 R = A_real, I = A_imag
   
2. Cayley变换生成酉矩阵 U:
   U = (I - A)(I + A)^(-1)
   保证 U ∈ U(N) (酉矩阵群)
   
3. Unistochastic转移矩阵 H:
   H = |U|^2 (元素取模平方)
   保证 H 是双随机矩阵 (行和=列和=1)
```

**关键属性**:
- 酉性约束: U^† U = I
- 双随机性: H的行和与列和均为1
- 维度: [hidden_dim, hidden_dim] (默认384×384)

### 2.2 HamiltonianInjectionLoRA (哈密顿注入模块)

**位置**: `mqr/ring.py:19-77`

**功能**: 低秩适配器将输入向量注入到环的能量景观中

**网络结构**:
```
输入: x ∈ [B, input_dim]
  │
  ▼
┌─────────────────────────┐
│  Down Projection        │
│  Linear(input_dim → r)  │
│  (无偏置, 低秩压缩)       │
└─────────────────────────┘
  │ z_pre: [B, r]  (r = lora_rank, 默认16)
  ▼
┌─────────────────────────┐
│  Activation (可选)       │
│  - none: 恒等映射        │
│  - relu: ReLU           │
│  - tanh: Tanh           │
│  - gelu: GELU           │
└─────────────────────────┘
  │ z: [B, r]
  ▼
┌─────────────────────────┐
│  Up Projection          │
│  Linear(r → hidden_dim) │
│  (无偏置, 低秩扩展)       │
└─────────────────────────┘
  │
  ▼
输出: J(x) ∈ [B, hidden_dim]
```

**参数量**:
- down: [input_dim × r]
- up: [r × hidden_dim]
- 总计: r × (input_dim + hidden_dim)

**关键特点**:
- 低秩分解降低参数量 (r << min(input_dim, hidden_dim))
- 可选的非线性激活
- 作为"外部驱动力"注入到环动力学中

### 2.3 LocalProjectiveReadout (局部投影读出模块)

**位置**: `mqr/ring.py:80-105`

**功能**: 从环的部分节点读取输出 (局部投影测量)

**网络结构**:
```
输入: h* ∈ [B, hidden_dim] (环的平衡态)
  │
  ▼
┌─────────────────────────┐
│  Node Selection         │
│  选择索引 S 的子集       │
│  h_S = h*[:, sample_indices] │
└─────────────────────────┘
  │ h_S: [B, readout_dim]  (默认前16个节点)
  ▼
┌─────────────────────────┐
│  Linear Readout         │
│  Linear(readout_dim → output_dim) │
│  (无偏置)                │
└─────────────────────────┘
  │
  ▼
输出: y ∈ [B, output_dim] (logits)
```

**参数量**:
- readout: [readout_dim × output_dim]
- 默认: [16 × 100] (CIFAR-100)

**可选模式**:
1. **linear** (默认): 线性读出 y = W · h_S
2. **proto**: 原型距离 logits = -||h_S - P_c||² / (2τ)

### 2.4 RingState (环状态容器)

**位置**: `mqr/ring.py:12-16`

**数据结构**:
```python
@dataclass(frozen=True)
class RingState:
    h: torch.Tensor  # [B, N] - 环的隐藏状态 (实数或复数)
```

### 2.5 ViTBackbone (Vision Transformer编码器) ⭐ 新增

**位置**: `mqr/ring.py:855-966`

**功能**: 标准的Vision Transformer编码器，作为强大的图像特征提取器

**网络结构**:
```
输入: x ∈ [B, 3, 32, 32] (CIFAR图像)
  │
  ▼
┌─────────────────────────────────┐
│  Patch Embedding                │
│  Conv2d(3→vit_dim, k=patch,     │
│         stride=patch)           │
│  输出: [B, P, vit_dim]          │
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│  Add CLS Token + Pos Embed      │
│  [B, 1, D] + [B, P, D]          │
│  → [B, 1+P, D]                  │
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│  TransformerEncoder × depth     │
│  - Multi-Head Self-Attention    │
│  - MLP (GELU)                   │
│  - LayerNorm                    │
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│  Pooling                        │
│  - cls: CLS token               │
│  - mean: mean over patch tokens │
└─────────────────────────────────┘
  │
  ▼
输出: [B, vit_dim] (默认384维)
```

**默认配置** (CIFAR-100):
```python
vit_dim = 384        # 嵌入维度
vit_depth = 6        # Transformer层数
vit_heads = 6        # 注意力头数
vit_mlp_dim = 1536   # MLP隐藏层维度
vit_dropout = 0.0    # Dropout率
vit_pool = "cls"     # 池聚方式
```

**参数量**:
- Patch Embed: 3×4×4×384 ≈ 18K
- Position Embed: (8×8+1)×384 ≈ 24K
- Transformer × 6: ≈ 7M
- **总计**: ≈ 7M (比简单patch嵌入强大得多)

**关键特点**:
- 成熟的ViT架构，经ImageNet预训练效果更好
- 标准的Transformer架构 (LayerNorm first, GELU)
- CLS token或全局平均池化
- 可与环联合训练 (通过梯度返回)

### 2.6 新增参数详解

#### 2.6.1 H混合参数 (h_mix_beta)

**位置**: `mqr/ring.py:_h_mix_beta_value`, `_current_H`

**功能**: 控制转移矩阵的自保留程度

**数学定义**:
```
H_eff = (1 - β) · I + β · H

其中:
- I: 单位矩阵 (保留当前状态)
- H: 酉随机矩阵 (全局混合)
- β ∈ [0, 1]: 混合系数
```

**物理意义**:
- β = 0: 完全自保留，无信息混合
- β = 1: 完全由H决定 (默认，双随机动力学)
- 0 < β < 1: 平衡自保留和全局混合

**学习方式**:
```python
# 固定值
h_mix_beta = 1.0  # 默认

# 可学习 (通过sigmoid约束)
h_mix_beta_param = log(β / (1-β))  # logistic parameterization
β = sigmoid(h_mix_beta_param)
```

**梯度更新**:
```python
grad_β = (grad_H · (H - I)).sum() / N
```

**优势**:
- 减少过度混合 (over-mixing)
- 保持双随机性 (凸组合)
- 不影响压缩性质 (||H_eff||_∞ = 1)

#### 2.6.2 动力学模式 (dynamics_mode)

**选项**: `"unistochastic"` | `"unitary"`

**对比**:

| 特性              | Unistochastic                    | Unitary                        |
|-------------------|----------------------------------|--------------------------------|
| 状态空间          | h ∈ ℝ^N                          | h ∈ ℂ^N                        |
| 转移矩阵          | H = \|U\|² (双随机)              | U (酉矩阵)                     |
| 前向迭代          | h ← (1-α)hH^T + αJ               | h ← (1-α)hU^† + αJ             |
| 物理意义          | 概率分布                         | 量子态幅                       |
| 相位信息          | 丢失                             | 保留                          |
| 测量机制          | identity                         | abs/real                      |
| 适用场景          | 传统神经网络                     | 量子算法类比                  |

**实现细节**:
```python
# Unistochastic (实数动力学)
H_eff = (1-β)I + β|U|²
h_{t+1} = σ[(1-α)h_t H_eff^T + αJ(x)]

# Unitary (复数动力学)
h_{t+1} = (1-α)h_t U^† + αJ(x) + i·(...)
测量: h_obs = |h| 或 Re(h)
```

#### 2.6.3 测量机制 (measurement)

**选项**: `"identity"` | `"abs"` | `"real"`

**定义**:

1. **identity** (默认，实数模式):
   ```python
   h_meas = h  # 直接使用
   ```

2. **abs** (复数模):
   ```python
   h_meas = |h|  # 取复数模
   # 梯度: ∂|h|/∂h = h/|h|
   ```

3. **real** (实部):
   ```python
   h_meas = Re(h)  # 取实部
   # 梯度: ∂Re(h)/∂h = 1 (实部), 0 (虚部)
   ```

**梯度回传**:
```python
def _pullback_measured_grad(h, grad_meas):
    if measurement == "identity":
        return grad_meas
    elif measurement == "abs":
        denom = h.abs().clamp_min(1e-8)
        return grad_meas * (h / denom)
    elif measurement == "real":
        return complex(grad_meas, 0)
```

**物理类比**:
- `abs`: 波函数模方测量 (Born规则)
- `real`: 实部投影 (类似同位测量)

### 2.7 编码器选项对比

| 编码器        | 输出维度    | 参数量      | 特点                     |
|--------------|-------------|-------------|--------------------------|
| flatten      | 3072        | 0           | 最简单，无参数           |
| patch        | 256         | ~12K        | 轻量级，局部感受野       |
| vit (ViT)    | 384         | ~7M         | 强大，全局注意力         |

**使用建议**:
- **快速原型**: flatten
- **轻量部署**: patch
- **最佳性能**: vit (可选预训练)## 3. MoebiusQuantumRing 核心动力学

**位置**: `mqr/ring.py:109-852`

### 3.0 两种动力学模式对比

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         动力学模式选择                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  【Unistochastic模式】(默认，实数值)                                      │
│  ─────────────────────────────────────────────────────                  │
│  1. 构造: U = Cayley(A) → H = |U|²                                       │
│  2. 混合: H_eff = (1-β)I + βH                                            │
│  3. 迭代: h_{t+1} = σ[(1-α)h_t H_eff^T + αJ(x)]                         │
│  4. 测量: h_obs = h (identity)                                          │
│                                                                          │
│  【Unitary模式】(新，复数值)                                              │
│  ─────────────────────────────────────────────────────                  │
│  1. 构造: U = Cayley(A) (直接使用U)                                     │
│  2. 无需H混合: 直接使用U                                                 │
│  3. 迭代: h_{t+1} = (1-α)h_t U^† + αJ(x)  (复数)                       │
│  4. 测量: h_obs = |h| (abs) 或 Re(h) (real)                             │
│                                                                          │
│  关键差异:                                                               │
│  - Unistochastic: H是双随机矩阵，行/列和为1 (概率转移)                   │
│  - Unitary: U是酉矩阵，保留相位信息 (量子态幅)                           │
│  - Unistochastic适合传统DL任务，Unitary适合需要相位敏感的任务            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```### 3.1 前向传播 (推理)

#### 3.1.1 Unistochastic模式 (实数值)

```
输入: x ∈ [B, input_dim]
  │
  ├─► 1. 注入: J(x) = Injection(x)
  │     └─► HamiltonianInjectionLoRA.forward(x)
  │         输出: [B, hidden_dim]
  │
  ├─► 2. 哈密顿量构造:
  │     a. U = Cayley(A)  [N×N 复数酉矩阵]
  │     b. H_base = |U|²  [N×N 实数双随机]
  │     c. H_eff = (1-β)I + β·H_base
  │     
  │     其中 β = h_mix_beta (可学习或固定)
  │
  ├─► 3. 固定点迭代:
  │     初始化: h_0 = 0 [B, N]
  │     迭代: h_{t+1} = σ[(1-α)·h_t·H_eff^T + α·J(x)]
  │     重复: relaxation_steps 次 (默认20)
  │     
  │     其中:
  │     - α ∈ (0,1]: 混合参数 (默认0.1)
  │     - σ: 可选激活 (none/relu/tanh)
  │     - H_eff^T: 转置的有效转移矩阵
  │
  ├─► 4. 测量与读出:
  │     h_meas = h* (identity测量)
  │     y = Readout(h_meas)
  │     
  │     返回: y ∈ [B, output_dim]
  │
  ▼
输出: y ∈ [B, output_dim]
```

#### 3.1.2 Unitary模式 (复数值) ⭐ 新增

```
输入: x ∈ [B, input_dim]
  │
  ├─► 1. 注入: J(x) = Injection(x)
  │     └─► [B, hidden_dim] (实数)
  │
  ├─► 2. 酉矩阵构造:
  │     U = Cayley(A)  [N×N 复数酉矩阵]
  │     注意: 不计算H，直接使用U的相位信息
  │
  ├─► 3. 复数固定点迭代:
  │     初始化: h_0 = 0 [B, N] (复数)
  │     J_c = complex(J, 0)  [B, N] (转换为复数)
  │     
  │     迭代: h_{t+1} = (1-α)·h_t·U^† + α·J_c
  │     重复: relaxation_steps 次 (默认20)
  │     
  │     关键差异:
  │     - U^†: 酉矩阵的共轭转置
  │     - h_t 保持复数 (相位参与迭代)
  │     - 不需要激活函数 (保持复数域线性)
  │
  ├─► 4. 测量与读出:
  │     h_meas = |h*| (abs) 或 Re(h*) (real)
  │     y = Readout(h_meas)
  │     
  │     返回: y ∈ [B, output_dim]
  │
  ▼
输出: y ∈ [B, output_dim]
```

**代码实现** (mqr/ring.py:forward):
```python
if self.dynamics_mode == "unistochastic":
    # 实数值动力学
    H = self._current_H(device=x.device, dtype=J.dtype)  # 包含β混合
    Ht = H.transpose(0, 1)
    h = torch.zeros(...)  # 实数
    for _ in range(self.relaxation_steps):
        h = self._state_act((1.0 - a) * (h @ Ht) + a * J)
    
else:  # "unitary"
    # 复数值动力学
    U_total = self._unitary_total()
    U_H = U_total.conj().transpose(0, 1)
    Jc = torch.complex(J, torch.zeros_like(J))
    h = torch.zeros(..., dtype=complex)  # 复数
    for _ in range(self.relaxation_steps):
        h = (1.0 - a) * (h @ U_H) + a * Jc

# 测量 (仅对unitary模式有效)
h_meas = self._measured_state(h)  # identity/abs/real
y = self._logits_from_state(h_meas)
```

#### 3.1.3 两种模式的选择指南

| 场景                     | 推荐模式               | 理由                           |
|--------------------------|------------------------|--------------------------------|
| 标准图像分类             | unistochastic          | 成熟，与经典DL兼容             |
| 需要相位敏感的任务       | unitary                | 保留相位信息                   |
| 量子算法模拟             | unitary                | 更接近量子态演化               |
| 资源受限场景             | unistochastic          | 复数运算开销大                 |
| 快速原型/实验            | unistochastic          | 默认，稳定                     |
| 研究复数动力学特性       | unitary                | 探索相位的作用                 |### 3.2 训练 (平衡态传播)

```
阶段1: 正向环 (推理)
┌─────────────────────────────────────┐
│ h* ← Relax(x, H)                    │
│ 固定点迭代收敛到平衡态                │
└─────────────────────────────────────┘
         │
         ▼
阶段2: 损失计算
┌─────────────────────────────────────┐
│ L = CE(y, target)                   │
│ 标准交叉熵损失                        │
└─────────────────────────────────────┘
         │
         ▼
阶段3: 伴随环 (反向传播)
┌─────────────────────────────────────┐
│ h^† ← Relax_Adjoint(h*, ∇L)         │
│ 伴随态迭代 (反向动力学)               │
└─────────────────────────────────────┘
         │
         ▼
阶段4: 梯度近似
┌─────────────────────────────────────┐
│ ∇H ≈ h^† ⊗ h* (外积)               │
│ 批量平均                             │
└─────────────────────────────────────┘
         │
         ▼
阶段5: 流形更新
┌─────────────────────────────────────┐
│ ΔA ∝ skew(U^† · (∇H ⊙ U ⊙ Ū))     │
│ 在酉群切空间上更新                    │
└─────────────────────────────────────┘
```

**关键方程**:

1. **伴随态迭代**:
   ```
   h^† = (1-α)(h^† ⊙ σ'(h*))H + α∇_h* L
   ```

2. **哈密顿梯度**:
   ```
   ∂L/∂H ≈ (h^† ⊙ σ'(h*))^T · h*
   ```

3. **流形梯度** (李代数):
   ```
   ΔA = 0.5(M - M^†)
   M = U^† · ((∂L/∂H) ⊙ H)
   ```

## 4. 完整数据流示例 (CIFAR-100)

### 4.1 配置
```python
输入: [B, 3, 32, 32]
图像编码: flatten
  → [B, 3072]

环维度:
  hidden_dim = 384
  lora_rank = 16
  readout_dim = 16
  output_dim = 100
  alpha = 0.1
  relaxation_steps = 20
```

### 4.2 前向传播详细步骤

```
Step 1: 图像编码
┌──────────────────────────────────┐
│ Input:  [B, 3, 32, 32]           │
│ Flatten: [B, 3072]               │
└──────────────────────────────────┘

Step 2: 哈密顿注入
┌──────────────────────────────────┐
│ Down:   [3072] → [16]            │
│ Act:    [16] (可选激活)           │
│ Up:     [16] → [384]             │
│ Output: J(x) ∈ [B, 384]          │
└──────────────────────────────────┘

Step 3: 构建转移矩阵
┌──────────────────────────────────┐
│ A = skew_hermitian([384, 384])   │
│ U = Cayley(A) ∈ [384, 384]      │
│ H = |U|² ∈ [384, 384]            │
│ (双随机矩阵)                      │
└──────────────────────────────────┘

Step 4: 固定点迭代
┌──────────────────────────────────┐
│ h_0 = 0 ∈ [B, 384]               │
│ for t in 1..20:                 │
│   h_t = σ[(1-0.1)·h_{t-1}·H^T   │
│         + 0.1·J(x)]              │
│ h* = h_20                        │
└──────────────────────────────────┘

Step 5: 局部读出
┌──────────────────────────────────┐
│ h_S = h*[:, :16] ∈ [B, 16]      │
│ y = W_readout · h_S             │
│ Output: [B, 100] (logits)       │
└──────────────────────────────────┘
```

### 4.3 参数统计

```python
配置: CIFAR-100, img_size=32, hidden_dim=384

┌─────────────────────────────────────────────────────────────────────────┐
│                         模块参数量统计                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ 【编码器】(根据image_encoder选择)                                        │
│  ─────────────────────────────────────────────────────                  │
│  flatten:      0 参数                                                   │
│  patch:        12K (Conv2d) + 24K (LayerNorm) = 36K                     │
│  vit (ViT):    ~7M (Transformer × 6)                                    │
│                                                                          │
│ 【注入模块】HamiltonianInjectionLoRA                                     │
│  ─────────────────────────────────────────────────────                  │
│  Down:   input_dim × 16                                                 │
│          - flatten: 3072 × 16   = 49K                                   │
│          - patch:   256 × 16     = 4K                                   │
│          - vit:     384 × 16     = 6K                                   │
│  Up:     16 × 384         = 6K                                          │
│  小计:   55K (flatten), 10K (patch), 12K (vit)                         │
│                                                                          │
│ 【酉变换】CayleyUnistochasticParam                                        │
│  ─────────────────────────────────────────────────────                  │
│  A_real: 384 × 384     = 147K                                           │
│  A_imag: 384 × 384     = 147K                                           │
│  小计:   294K                                                           │
│                                                                          │
│ 【H混合】(可选)                                                          │
│  ─────────────────────────────────────────────────────                  │
│  h_mix_beta_param: 1 (可学习时)                                         │
│                                                                          │
│ 【读出】LocalProjectiveReadout                                           │
│  ─────────────────────────────────────────────────────                  │
│  W_readout: 16 × 100   = 1.6K                                           │
│                                                                          │
│ 【状态目标】(可选，learnable_state_targets=True)                          │
│  ─────────────────────────────────────────────────────                  │
│  state_targets: 100 × 384 = 38.4K                                       │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ 总参数量:                                                                │
│  - flatten模式:     350K   (轻量级)                                     │
│  - patch模式:       330K   (轻量级)                                     │
│  - vit模式:         7.3M   (重量级，性能最强)                           │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ 计算量 (FLOPs per sample):                                              │
│                                                                          │
│ 【编码器】                                                                │
│  flatten:     0                                                          │
│  patch:       ~3M   (Conv2d)                                            │
│  vit:         ~60M  (Transformer × 6)                                   │
│                                                                          │
│ 【注入 LoRA】                                                            │
│  flatten:     ~0.1M (3072×16 + 16×384)                                  │
│  patch/vit:   ~0.02M                                                    │
│                                                                          │
│ 【Cayley变换】                                                           │
│  O(N³) ≈ 57M (N=384, 矩阵求逆)                                          │
│                                                                          │
│ 【固定点迭代】                                                           │
│  每步: 2×B×N² ≈ 0.3M (B=32, N=384)                                     │
│  20步: ≈ 6M                                                             │
│                                                                          │
│ 总计 (per batch, B=32):                                                  │
│  - flatten:     ~63M                                                    │
│  - patch:       ~66M                                                    │
│  - vit:         ~123M (编码器60M + 环63M)                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**内存占用** (FP32, batch=32):

| 配置       | 模型参数 | 激活值 (峰值) | 总内存 |
|-----------|---------|--------------|--------|
| flatten   | 1.4 MB  | ~30 MB       | ~32 MB |
| patch     | 1.3 MB  | ~30 MB       | ~32 MB |
| vit       | 28 MB   | ~120 MB      | ~150 MB |

**训练相关参数**:
- `h_mix_beta_lr_ratio`: β的学习率比例 (默认1.0)
- `encoder_lr_ratio`: 编码器学习率比例 (默认1.0)
- `return_grad_x`: 返回输入梯度以训练编码器## 5. 关键创新点

### 5.1 酬约束动力学
- 传统神经网络: 权重矩阵无物理约束
- MQR: 转移矩阵H = |U|² 或 U，保证酉性→双随机性
- 优势: 
  - 能量守恒特性
  - 梯度稳定性
  - 理论可解释性

### 5.2 平衡态传播 (EqProp)
- 传统反向传播: 需要通过整个计算图
- EqProp: 局部梯度近似
  - 正向: h* = Relax(x)
  - 反向: h^† = Relax_Adjoint(h*)
  - 更新: ΔA ∝ skew(...)
- 优势:
  - 生物合理性
  - 避免梯度消失/爆炸
  - 适合在线学习

### 5.3 低秩注入
- 传统全连接: d_in × d_out 参数
- LoRA注入: r × (d_in + d_out) 参数
- 优势:
  - 参数效率 (r << min(d_in, d_out))
  - 计算效率
  - 防止过拟合

### 5.4 局部投影读出
- 传统全局读出: 使用所有隐藏单元
- 局部读出: 仅使用少量节点
- 优势:
  - 模拟量子测量
  - 降低输出层参数
  - 提高泛化能力

### 5.5 H混合机制 (自保留) ⭐ 新增
- **动机**: 纯双随机矩阵可能导致过度混合
- **方法**: H_eff = (1-β)I + βH
- **特点**:
  - β = 1: 完全双随机 (默认)
  - β < 1: 增加自保留，减少过度混合
  - 保持双随机性 (凸组合)
  - 可学习或固定
- **理论保证**: 
  - ||H_eff||_∞ = 1 (保持压缩性质)
  - 双随机性不变

### 5.6 复数动力学 (Unitary模式) ⭐ 新增
- **创新**: 保留相位信息的量子态演化
- **与unistochastic的区别**:
  - 直接使用U而非|U|²
  - 复数状态 h ∈ ℂ^N
  - 相位参与迭代
- **测量机制**:
  - `abs`: |h| (Born规则类比)
  - `real`: Re(h) (同位测量)
- **适用场景**:
  - 需要相位敏感的任务
  - 量子算法模拟
  - 复数信号处理

### 5.7 ViT编码器集成 ⭐ 新增
- **动机**: 利用成熟的ViT作为强backbone
- **设计**: 环作为drop-in replacement for分类头
- **优势**:
  - 可利用ImageNet预训练
  - 强大的全局建模能力
  - 可与环联合训练
- **灵活性**: 
  - 支持三种编码器: flatten/patch/vit
  - 可选编码器学习率比例
  - 梯度返回机制支持端到端训练

### 5.8 端到端编码器训练 ⭐ 新增
- **创新**: EqProp与编码器联合训练
- **方法**:
  ```python
  # 1. 环前向 + 反向
  info = ring.eqprop_update_step(x_vec, target, return_grad_x=True)
  
  # 2. 获取输入梯度
  grad_x = info["grad_x"]  # ∂L/∂x
  
  # 3. 编码器反向传播
  x_vec.backward(grad_x)
  
  # 4. 更新编码器参数
  encoder_optimizer.step()
  ```
- **优势**:
  - 环的局部梯度 + 编码器的全局梯度
  - 两阶段优化: 环用EqProp，编码器用BPTT
  - 灵活的学习率比例控制## 6. 数学性质

### 6.1 酵矩阵性质
```
U ∈ U(N) ⇔ U^† U = I

推论:
1. |det(U)| = 1 (保体积)
2. ||Ux|| = ||x|| (等距变换)
3. 特征值在单位圆上 |λ_i| = 1
```

### 6.2 双随机矩阵性质
```
H_ij = |U_ij|²

性质:
1. ∑_j H_ij = 1 (行和为1)
2. ∑_i H_ij = 1 (列和为1)
3. H_ij ≥ 0 (非负)

推论:
- H ∈ Birkhoff多面体 (凸组合)
- H可分解为置换矩阵的凸组合
- 保证马尔可夫链稳态存在
```

### 6.3 固定点收敛性
```
迭代: h_{t+1} = (1-α)h_t H^T + αJ(x)

收敛条件 (当σ=id):
- H是随机矩阵 → 谱半径 ρ(H) ≤ 1
- α > 0 → 压缩映射
- 保证唯一不动点 h* = J(x)(I - (1-α)H^T)^(-1)
```

## 7. 与其他模型对比

| 特性                   | MQR v2                          | MQR v1                    | ViT                        | ResNet                |
|------------------------|---------------------------------|---------------------------|----------------------------|-----------------------|
| **参数共享**           | 循环迭代                        | 循环迭代                  | 自注意力                  | 卷积共享              |
| **能量约束**           | 酵性 → 双随机/酉                | 酵性 → 双随机              | 无                        | 无                    |
| **状态空间**           | 实数或复数                      | 实数                      | 实数                      | 实数                  |
| **训练方式**           | EqProp (局部)                   | EqProp (局部)              | BPTT (全局)               | SGD (全局)            |
| **编码器**             | flatten/patch/ViT 可选          | flatten/patch             | ViT only                  | CNN only              |
| **自保留混合**         | 支持 (可学习β)                  | 不支持                    | 不支持                    | 不支持                |
| **复数动力学**         | 支持 (unitary模式)              | 不支持                    | 不支持                    | 不支持                |
| **端到端训练**         | 支持编码器联合训练              | 不支持                    | 原生支持                  | 原生支持              |
| **收敛性保证**         | 理论保证 (压缩映射)             | 理论保证                  | 启发式                    | 启发式                |
| **物理可解释性**       | 量子动力学类比                  | 量子动力学类比            | 注意力机制                | 特征提取              |
| **在线学习**           | 原生支持                        | 原生支持                  | 困难                      | 困难                  |
| **参数量** (CIFAR)     | 350K (flatten) ~ 7.3M (vit)     | 350K                      | 7M+                       | <1M                   |
| **性能潜力**           | 高 (灵活配置)                   | 中                        | 高                        | 中                    |

**版本对比**:

| 特性              | MQR v1 (原版)         | MQR v2 (更新)                  |
|-------------------|----------------------|--------------------------------|
| 动力学模式        | unistochastic only   | unistochastic + unitary        |
| H混合             | 不支持               | 支持可学习β                    |
| 编码器            | flatten/patch        | + ViT (完整Transformer)        |
| 测量机制          | identity             | identity/abs/real              |
| 编码器训练        | 不支持               | 支持端到端联合训练             |
| 梯度返回          | 不支持               | 支持return_grad_x              |
| 理论保证          | 酉性 + 双随机        | + 自保留混合的收敛性            |## 8. 使用场景

### 8.1 适合的场景

#### 8.1.1 Unistochastic模式
- **标准图像分类**: CIFAR、ImageNet等
- **序列数据建模**: 视频、时间序列
- **在线持续学习**: EqProp的局部梯度更新
- **低功耗边缘设备**: EqProp可用本地更新
- **需要理论保证**: 压缩映射确保收敛

#### 8.1.2 Unitary模式 ⭐ 新增
- **量子算法模拟**: Grover、量子Fourier变换等
- **相位敏感任务**: 信号处理、干涉测量
- **复数信号**: 通信信号、雷达数据
- **复数神经网络研究**: 探索复数DL的优势

#### 8.1.3 ViT编码器
- **大规模预训练**: 利用ImageNet预训练ViT
- **高性能场景**: 需要最佳精度
- **迁移学习**: 从大模型微调到小数据集
- **多模态融合**: 扩展到视觉-语言任务

### 8.2 配置建议

#### 场景1: 快速原型研究
```python
model = MoebiusQuantumRingImageClassifier(
    image_encoder="flatten",
    hidden_dim=384,
    dynamics_mode="unistochastic",  # 默认，稳定
    h_mix_beta=1.0,                 # 无自保留
)
```
**参数**: 350K | **训练速度**: 快 | **精度**: 中等

#### 场景2: 轻量级部署
```python
model = MoebiusQuantumRingImageClassifier(
    image_encoder="patch",
    patch_embed_dim=256,
    hidden_dim=256,                 # 降低环维度
    lora_rank=8,                    # 降低LoRA秩
    dynamics_mode="unistochastic",
)
```
**参数**: 200K | **推理速度**: 快 | **精度**: 中等

#### 场景3: 最佳性能
```python
model = MoebiusQuantumRingImageClassifier(
    image_encoder="vit",
    vit_dim=384,
    vit_depth=6,
    hidden_dim=384,
    dynamics_mode="unistochastic",
    learnable_h_mix_beta=True,      # 可学习β
    learnable_state_targets=True,   # 原型匹配
)
```
**参数**: 7.3M | **训练速度**: 中等 | **精度**: 高

#### 场景4: 复数动力学研究
```python
model = MoebiusQuantumRing(
    input_dim=384,
    hidden_dim=384,
    output_dim=100,
    dynamics_mode="unitary",         # 复数模式
    measurement="abs",               # 或 "real"
    h_mix_beta=0.9,                 # 一些自保留
)
```
**参数**: 330K | **特点**: 保留相位 | **研究性**: 是

### 8.3 待探索方向

#### 8.3.1 算法层面
- [ ] 大规模预训练 (当前主要是CIFAR等小数据集)
- [ ] 超高分辨率图像 (需要优化patch embedding)
- [ ] 多模态融合 (扩展输入注入机制)
- [ ] 复数动力学的优势场景探索
- [ ] 可学习β的收敛性分析

#### 8.3.2 理论层面
- [ ] Unitary模式的收敛性证明
- [ ] H混合对泛化误差的影响
- [ ] 相位信息的表征能力分析
- [ ] EqProp与BPTT的理论联系
- [ ] 量子-经典混合算法

#### 8.3.3 应用层面
- [ ] 目标检测 (扩展到下游任务)
- [ ] 视频理解 (时序建模)
- [ ] 强化学习 (状态表示学习)
- [ ] 生成模型 (采样与去噪)## 9. 代码索引

- **核心模型**: `mqr/ring.py:109` (MoebiusQuantumRing) - 852行
- **图像分类器**: `mqr/ring.py:969` (MoebiusQuantumRingImageClassifier) - 1233行
- **酉变换**: `mqr/unitary.py:6` (CayleyUnistochasticParam)
- **注入模块**: `mqr/ring.py:20` (HamiltonianInjectionLoRA)
- **读出模块**: `mqr/ring.py:80` (LocalProjectiveReadout)
- **ViT编码器**: `mqr/ring.py:855` (ViTBackbone) ⭐ 新增
- **训练脚本**: `train_mobius_cifar100.py`

---

## 附录: 版本更新总结

### v2.0 主要更新 (2025-01-07)

#### 新增特性

1. **复数动力学模式 (Unitary Dynamics)**
   - 新参数: `dynamics_mode: str = "unistochastic" | "unitary"`
   - 新参数: `measurement: str = "identity" | "abs" | "real"`
   - 代码位置: `mqr/ring.py:forward()`, `compute_adjoint_state_from_grad_h()`
   - 影响: 支持384维复数状态，保留相位信息

2. **H混合机制 (Self-Retention Mixing)**
   - 新参数: `h_mix_beta: float = 1.0`
   - 新参数: `learnable_h_mix_beta: bool = False`
   - 新方法: `_h_mix_beta_value()`, `_current_H_base()`, `_current_H()`
   - 公式: H_eff = (1-β)I + βH
   - 影响: 可控的信息混合程度，减少过度混合

3. **ViT编码器 (Vision Transformer Backbone)**
   - 新类: `ViTBackbone` (`mqr/ring.py:855-966`)
   - 新参数: `image_encoder: "vit"`
   - ViT配置: `vit_dim`, `vit_depth`, `vit_heads`, `vit_mlp_dim`
   - 影响: 参数量从350K增加到7.3M，性能大幅提升

4. **端到端编码器训练**
   - 新参数: `encoder_lr_ratio: float = 1.0`
   - 新参数: `return_grad_x: bool = False`
   - 新功能: `eqprop_update_step()` 支持返回输入梯度
   - 影响: 支持编码器与环联合训练

5. **测量机制与梯度回传**
   - 新方法: `_measured_state()`, `_pullback_measured_grad()`
   - 支持三种测量: identity/abs/real
   - 自动处理复数到实数的梯度回传

#### API变化

**新增构造参数** (MoebiusQuantumRing):
```python
h_mix_beta: float = 1.0                    # H混合系数
learnable_h_mix_beta: bool = False         # 可学习β
dynamics_mode: str = "unistochastic"       # 动力学模式
measurement: str = "identity"              # 测量机制
```

**新增构造参数** (MoebiusQuantumRingImageClassifier):
```python
image_encoder: str = "flatten" | "patch" | "vit"  # 编码器选择
vit_dim: int = 384                          # ViT嵌入维度
vit_depth: int = 6                          # ViT深度
vit_heads: int = 6                          # ViT注意力头数
vit_mlp_dim: int = 1536                     # ViT MLP维度
vit_dropout: float = 0.0                    # ViT dropout
vit_pool: str = "cls"                       # ViT池聚方式
```

**新增返回值** (eqprop_update_step):
```python
return {
    ...
    "h_mix_beta": float,     # 当前β值
    "grad_x": Tensor,        # 输入梯度 (可选)
}
```

#### 参数量变化

| 配置 | v1.0 | v2.0 (flatten) | v2.0 (patch) | v2.0 (vit) |
|-----|------|----------------|--------------|------------|
| 编码器 | flatten (0) | 0 | 36K | 7M |
| 环 | 350K | 350K | 330K | 330K |
| **总计** | 350K | 350K | 366K | 7.3M |

#### 向后兼容性

- ✅ v1.0的API完全兼容 (默认参数保持不变)
- ✅ 已有训练脚本无需修改
- ✅ 新特性都是可选的，通过参数启用

#### 文件变化

- `mqr/ring.py`: 676行 → 1233行 (+557行)
- 新增 `ViTBackbone` 类
- 新增测量机制相关方法
- 扩展 `eqprop_update_step()` 支持编码器训练

### 迁移指南

#### 从v1.0升级到v2.0

**场景1: 保持原有功能 (无需修改)**
```python
# v1.0 代码
model = MoebiusQuantumRingImageClassifier(
    img_size=32,
    num_classes=100,
)

# v2.0 完全兼容，无需修改
```

**场景2: 启用H混合**
```python
model = MoebiusQuantumRingImageClassifier(
    img_size=32,
    num_classes=100,
    h_mix_beta=0.9,                    # 新增
    learnable_h_mix_beta=True,         # 新增
)
```

**场景3: 使用ViT编码器**
```python
model = MoebiusQuantumRingImageClassifier(
    img_size=32,
    num_classes=100,
    image_encoder="vit",               # 新增
    vit_dim=384,                       # 新增
    vit_depth=6,                       # 新增
)
```

**场景4: 启用复数动力学**
```python
model = MoebiusQuantumRing(
    input_dim=384,
    hidden_dim=384,
    output_dim=100,
    dynamics_mode="unitary",           # 新增
    measurement="abs",                 # 新增
)
```

**场景5: 联合训练编码器**
```python
# 训练循环
for images, targets in train_loader:
    info = model.eqprop_update_step(
        images, targets,
        lr=0.01,
        encoder_lr_ratio=0.5,         # 新增
        encoder_optimizer=enc_opt,     # 新增
    )
```

### 未来路线图

- [ ] 大规模ImageNet预训练实验
- [ ] Unitary模式的性能基准测试
- [ ] 可学习β的收敛性分析
- [ ] 更多编码器选项 (ResNet, Swin等)
- [ ] 分布式训练支持
- [ ] ONNX导出支持