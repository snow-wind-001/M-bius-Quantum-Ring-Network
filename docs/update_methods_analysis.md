# 酉矩阵更新方法的数学分析与比较

## 1. 背景：Möbius Quantum Ring 中的参数更新

在MQR架构中，核心是通过Cayley变换从反埃尔米特矩阵A生成酉矩阵U：
$$U = (I - A)(I + A)^{-1}, \quad A^\dagger = -A$$

然后通过 $H = |U|^2$（元素级取模平方）得到双随机矩阵H用于推理。

训练目标是通过梯度信息更新U（或等价地更新A），使损失函数L最小化。

---

## 2. 方法一：指数映射 (Exponential Map on U(n))

### 2.1 公式
$$U_{new} = \exp(-i \eta \Omega) \cdot U_{old}$$

其中 $\Omega$ 是从损失梯度导出的埃尔米特矩阵（可理解为"势能"）。

### 2.2 数学背景
- **李群视角**：酉群U(n)是一个李群，指数映射 $\exp: \mathfrak{u}(n) \to U(n)$ 将李代数元素映射回群上。
- **测地线**：对于酉矩阵流形，$U(t) = \exp(-it\Omega) U_0$ 是一条测地线。
- **物理解释**：这模拟了量子力学中薛定谔方程的演化，$\Omega$ 扮演"哈密顿量"的角色。

### 2.3 计算要求
1. 需要计算 $\Omega = f(\nabla_H L, U)$：从H的梯度推导出李代数元素
2. 需要计算**矩阵指数** $\exp(-i \eta \Omega)$
3. 矩阵乘法 $\exp(-i \eta \Omega) \cdot U_{old}$

### 2.4 优点
- **几何优美**：直接在流形上沿测地线移动
- **物理直觉强**：类似量子演化
- **保证U_{new}仍是酉矩阵**：指数映射天然保持群结构

### 2.5 缺点
- **计算昂贵**：矩阵指数需要 $O(n^3)$ 复杂度（通常通过Padé逼近或特征分解）
- **不直接与Cayley参数化兼容**：如果使用Cayley参数化A，还需要额外步骤从U_new恢复A_new
- **数值稳定性**：对于大学习率可能不稳定

---

## 3. 方法二：李代数参数更新 (ΔA Update via Cayley)

### 3.1 公式
$$\Delta A \propto \text{skew}\left( U^\dagger \cdot \left( \frac{\partial L}{\partial H} \odot U \odot \bar{U} \right) \right)$$

其中 $\text{skew}(M) = \frac{1}{2}(M - M^\dagger)$ 提取反埃尔米特部分。

### 3.2 数学推导

从 $H_{ij} = |U_{ij}|^2 = U_{ij} \bar{U}_{ij}$ 出发：

**Step 1**: 计算 $\frac{\partial L}{\partial U}$ 从 $\frac{\partial L}{\partial H}$

$$\frac{\partial L}{\partial U_{ij}} = \frac{\partial L}{\partial H_{ij}} \cdot \frac{\partial H_{ij}}{\partial U_{ij}} = \frac{\partial L}{\partial H_{ij}} \cdot \bar{U}_{ij}$$

所以 $\nabla_U L = \nabla_H L \odot \bar{U}$

**Step 2**: 映射到李代数（切空间）

酉群U(n)在点U处的切空间是 $T_U U(n) = \{U \cdot \Xi : \Xi^\dagger = -\Xi\}$

将欧几里得梯度投影到切空间：
$$\text{grad}_U L = U \cdot \text{skew}(U^\dagger \nabla_U L)$$

**Step 3**: 用Cayley参数化表示

因为 $U = (I-A)(I+A)^{-1}$，我们需要找到 $\Delta A$ 使得更新后的A仍然是反埃尔米特的。

关键洞察：
$$\Delta A \propto \text{skew}(U^\dagger \cdot (\nabla_H L \odot U \odot \bar{U}))$$

这里 $\nabla_H L \odot H = \nabla_H L \odot |U|^2 = \nabla_H L \odot U \odot \bar{U}$

### 3.3 计算要求
1. 计算 $\nabla_H L \odot H$ (Hadamard乘积)
2. 矩阵乘法 $U^\dagger \cdot (...)$
3. 提取反埃尔米特部分 $\text{skew}(\cdot)$
4. 简单的参数加法更新 $A \leftarrow A - \eta \Delta A$

### 3.4 优点
- **计算高效**：仅需矩阵乘法和加法，无需矩阵指数
- **与Cayley参数化天然兼容**：直接更新A
- **数值稳定**：小学习率下的线性近似
- **实现简洁**：公式明确无歧义

### 3.5 缺点
- **是一阶近似**：相比测地线，是在切空间中的线性近似
- **大学习率时可能偏离流形**：需要Cayley变换"拉回"到群上

---

## 4. 两种方法的核心等价性分析

### 4.1 在小学习率极限下

当 $\eta \to 0$ 时，两种方法趋于等价：

**指数映射**：
$$U_{new} = \exp(-i\eta\Omega) U_{old} \approx (I - i\eta\Omega) U_{old}$$

**Cayley with ΔA**：
$$U_{new} = (I - A_{new})(I + A_{new})^{-1}$$

其中 $A_{new} = A_{old} + \Delta A$

当 $\Delta A$ 很小时，可以展开：
$$U_{new} \approx U_{old} + 2(I + A_{old})^{-1} \Delta A (I + A_{old})^{-1}$$

### 4.2 关键差异

| 特性 | 指数映射 | ΔA 更新 |
|------|----------|---------|
| 计算复杂度 | $O(n^3)$ (矩阵指数) | $O(n^3)$ (矩阵乘法) |
| 常数因子 | 大（Padé逼近等） | 小 |
| 酉性保证 | 精确 | 通过Cayley变换精确 |
| 实现复杂度 | 高 | 低 |
| 与Cayley兼容 | 需要额外步骤 | 天然兼容 |

---

## 5. 结论与选择建议

### 5.1 选择ΔA更新方法的理由

1. **计算效率**：无需计算矩阵指数，仅需标准矩阵运算
2. **实现简洁**：公式明确，易于调试和验证
3. **与架构一致**：我们已经使用Cayley参数化，ΔA更新与之完美契合
4. **数值稳定**：Cayley变换天然保证输出是酉矩阵，无论ΔA多大
5. **文档支持**：HTML中明确给出了ΔA的完整公式

### 5.2 理论正当性

虽然ΔA是一阶近似，但：
- 在实际深度学习中，小学习率是标准做法
- Cayley变换在每次更新后"重新投影"到酉群，消除累积误差
- 这类似于"重参数化梯度下降"，在流形优化中被广泛接受

### 5.3 最终选择

**我们选择 ΔA 更新方法**，因为它：
- 更明确、无歧义
- 计算更高效
- 与我们的Cayley参数化天然兼容
- 在HTML文档中有完整的数学推导

---

## 6. 实现验证

当前 `mqr/ring.py` 中的 `eqprop_update_step` 方法实现：

```python
# 4.3 Unitary manifold update (core "phase / Lie algebra" step)
lr_u = lr * unitary_lr_ratio
if lr_u > 0:
    U = self.unitary_param.unitary()  # [N, N] complex
    H_u = U.abs().pow(2)  # [N, N] real

    # HTML: ΔA ∝ skew( U^† · ( (∂L/∂H) ⊙ U ⊙ Ū ) )
    inner = (grad_H * H_u).to(dtype=U.dtype)  # (∂L/∂H) ⊙ H = (∂L/∂H) ⊙ U ⊙ Ū
    M = U.conj().transpose(-2, -1) @ inner     # U^† · inner
    delta_A = 0.5 * (M - M.conj().transpose(-2, -1))  # skew-Hermitian

    # 更新 A 的实部和虚部参数
    self.unitary_param.A_real.data.add_(delta_A.real, alpha=-lr_u)
    self.unitary_param.A_imag.data.add_(delta_A.imag, alpha=-lr_u)
```

这与HTML文档中的公式完全一致。
