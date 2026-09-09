# Möbius Quantum Ring：严格数学审计、在线正交学习与 2B 模型可行性

> 审计对象：当前 `mqr/unitary.py`、`mqr/ring.py`、训练脚本、测试、论文与已有 checkpoint。
> 结论边界：本文可以严格证明结构约束、固定点收敛、隐式梯度和局部抗遗忘性质；不能用数学证明替代“优于 LoRA”“具备动物的一般学习能力”等必须由实验回答的命题。

## 1. 结论先行

| 问题 | 结论 | 置信度 |
|---|---|---|
| Cayley 参数化是否始终保酉 | 是（浮点误差除外） | 严格证明 |
| `H=|U|²` 是否双随机 | 是 | 严格证明 |
| 阻尼环是否收敛 | `0<α≤1` 且状态激活 1-Lipschitz 时唯一收敛 | 严格证明 |
| 手工反向是否可等于真梯度 | 修正后，在正/反固定点收敛时等于隐式梯度 | 证明 + 数值相对误差 `<1e-15` |
| 酉参数是否自动防灾难性遗忘 | 否；酉性只约束状态传播 | 反例成立 |
| 线性环是否比 LoRA 更有表达力 | 否；精确等价于一个秩不超过 `r` 的 LoRA | 严格证明 |
| 非线性多环是否有额外价值 | 可能；代价是迭代时延与更难的持续学习 | 待实验 |
| 挂到 2B 模型是否工程可行 | 小环、稀疏路由、缓存连接矩阵时可行 | 参数/复杂度闭合 |
| 是否足以模拟一般动物学习 | 不能；最多先验证快速联想、短时状态和受控持续学习 | 当前证据不足 |

本次审计还发现：历史更新式把酉群切向量直接当成 Cayley 坐标梯度，并误写了 `∂|U|²/∂U`。旧数值脚本又把“参数变化量与正梯度同向”误判为下降。重构的验证在 30 个随机问题上显示旧方向平均余弦约 `-0.259`，27/30 与真梯度负相关。当前代码已改为精确 Cayley 拉回。

## 2. 当前算法的精确定义

令环宽为 `N`，输入维度为 `d`，低秩为 `r`。当前实现构造

\[
A=\tfrac12[(R-R^\top)+i(S+S^\top)],\qquad A^\dagger=-A,
\]

\[
U=(I-A)(I+A)^{-1},\qquad H_{ij}=|U_{ij}|^2.
\]

可选冻结基矩阵时，`U_total=U_policy U_base`。自保持连接为

\[
H_\beta=(1-\beta)I+\beta H,\qquad 0\le\beta\le1.
\]

注入与实环动力学为

\[
J(x)=W_{up}\phi(W_{down}x),
\]

\[
h_{t+1}=\sigma\!\left(qH_\beta h_t+\alpha J(x)\right),
\qquad q=1-\alpha,
\]

这里采用列向量记法；代码中的行向量写法是 `h @ H_beta.T`。复数模式改为 `h_{t+1}=qUh_t+αJ` 的等价行向量形式，并在读出前测量 `|h|` 或 `Re(h)`。读出只采样节点集合 `S`，使用线性层或原型距离。

## 3. 结构与收敛的严格证明

### 定理 3.1：反 Hermitian 构造

因为 `(R-Rᵀ)ᵀ=-(R-Rᵀ)` 且 `(S+Sᵀ)ᵀ=S+Sᵀ`，直接取共轭转置得到 `A†=-A`。原始 `R,S` 共存 `2N²` 个实参数，但投影后的 `A` 只有 `N²` 个实自由度，因此当前存储本身有冗余。

### 定理 3.1a：式 (13) 完整覆盖 \(\mathfrak u(N)\)，但旧坐标过完备

任意 $A\in\mathfrak u(N)$ 可唯一写成 $A=X+iY$，其中 $X,Y$ 为实矩阵。
由 $A^\dagger=-A$ 得

\[
X^T=-X,\qquad Y^T=Y.
\]

对映射

\[
\Psi(R,S)=\tfrac12[(R-R^T)+i(S+S^T)]
\]

取 $R=X,S=Y$，立即有 $\Psi(R,S)=A$。因此式 (13) 对
\(\mathfrak u(N)\) 是满射，不会漏掉一整族 Lie 代数方向。它的问题是不可辨识：
任意实对称增量可加到 $R$，任意实反对称增量可加到 $S$，而 $A$ 不变。
这两个核空间的维数之和为

\[
\frac{N(N+1)}2+\frac{N(N-1)}2=N^2.
\]

所以 $2N^2$ 个旧参数的 Jacobian 秩恰为 $N^2=\dim_\R\mathfrak u(N)$。
代码中的 `minimal` 模式分别用实反对称和虚对称的 Frobenius 正交基存储
\(N(N-1)/2+N(N+1)/2=N^2\) 个坐标；`projected` 模式仅为旧 checkpoint
兼容而保留。

Cayley 微分 $dU=-2(I+A)^{-1}(dA)(I+A)^{-1}$ 是两个可逆矩阵对 $dA$
的双边乘法，所以对每个有限 $A$ 都是实线性同构。Cayley 本身不会丢失切向
梯度。其像是开集

\[
\{U\in U(N):\det(I+U)\neq0\},
\]

而非整个 $U(N)$；遗漏的是谱含 $-1$ 的酉矩阵。

### 定理 3.2：Cayley 变换保酉

反 Hermitian 矩阵的特征值均为纯虚数，所以 `-1` 不可能是 `A` 的特征值，`I+A` 对所有合法 `A` 都可逆。又因为 `I-A` 与 `I+A` 都是 `A` 的多项式，二者可交换：

\[
U^\dagger=(I-A)^{-1}(I+A),
\]

\[
U^\dagger U=(I-A)^{-1}(I+A)(I-A)(I+A)^{-1}=I.
\]

应注意：Cayley 映射遗漏的是“含特征值 `-1` 的酉矩阵”，不是“含特征值 `-1` 的 A”。

### 定理 3.2a：Cayley 与模平方的复合完整覆盖 unistochastic 集

令 \(\mathcal C=\{(I-A)(I+A)^{-1}:A^\dagger=-A\}\)。虽然
\(\mathcal C\neq U(N)\)，但

\[
\boxed{\{|U|^2:U\in\mathcal C\}=\mathcal U_N},
\]

其中 \(\mathcal U_N\) 是全部 unistochastic 矩阵。右侧包含左侧由定义立即成立。
反过来，任取 \(H\in\mathcal U_N\)，存在 \(V\in U(N)\) 使
\(H=|V|^2\)。记 \(V\) 的特征值为 \(\lambda_1,\ldots,\lambda_N\)。对单位复数
\(\zeta\)，仅当

\[
\zeta\in F=\{-\lambda_1^{-1},\ldots,-\lambda_N^{-1}\}
\]

时，\(\zeta V\) 才含特征值 \(-1\)。禁集 \(F\) 至多含 \(N\) 个点，因此可选
\(\zeta\notin F\)。此时

\[
A=(I-\zeta V)(I+\zeta V)^{-1}
\]

是有限斜 Hermitian 矩阵，且 Cayley\((A)=\zeta V\)。又因全局相位不改变元素
模方，\(|\zeta V|^2=|V|^2=H\)。实现使用多于 \(N\) 个不同候选相位，并选择
\(\sigma_{\min}(I+\zeta V)\) 最大者；鸽巢原理保证至少一个候选不在 \(F\)。

该定理只消除了 Cayley 的 \(-1\) 谱遗漏对 **复合转移集合** 的影响。它不把
unistochastic 集扩张为整个 Birkhoff 多面体，也不保证
\(U\mapsto|U|^2\) 在每个点局部满秩。靠近禁谱时逆 Cayley 坐标仍可能病态，
所以实现返回所选相位的最小奇异值，训练仍需监控条件性。

### 推论 3.2b：Cayley 与 unistochastic 转移的有限漂移界

对任意斜 Hermitian (A,B)，令 (R_A=(I+A)^{-1})。因为 (A)
正规且谱为纯虚数，(R_A) 的谱范数不超过 1。利用
(U(A)=2R_A-I) 和解析恒等式

\[
R_A-R_B=R_A(B-A)R_B,
\]

可得

\[
\boxed{\|U(A)-U(B)\|_F\le2\|A-B\|_F}.
\]

酉矩阵的每个元素模不超过 1，所以

\[
\big||u|^2-|v|^2\big|
\le (|u|+|v|)|u-v|\le2|u-v|.
\]

对全部元素求 Frobenius 范数即得

\[
\boxed{\||U(A)|^2-|U(B)|^2\|_F\le4\|A-B\|_F}.
\]

因此小环上的 (A) trust region 可给出保守的 (H) 漂移预算。
这不控制读出、注入或冻结主干的功能漂移，后者仍需要独立
constancy probe。

### 定理 3.3：模方矩阵双随机

`H_ij≥0`。由 `UU†=I`，每一行的模方和为 1；由 `U†U=I`，每一列的模方和为 1。因此 `H` 双随机。`I` 与 `H` 的凸组合 `H_β` 仍双随机。

进一步，所有双随机矩阵都满足

\[
\|H\|_1=\|H\|_\infty=1,\qquad \|H\|_2=1.
\]

最后一个等号对所有双随机矩阵成立，并非仅置换矩阵成立：`H1=1` 给出 `||H||₂≥1`，而 `||H||₂≤sqrt(||H||₁||H||∞)=1`。`H` 对一般向量只是不扩张；它仅对非负向量保持 `L1` 总量，并不普遍保持欧氏“能量”。真正保持 `L2` 的是 `U`。

### 命题 3.3a：表示瓶颈来自模平方映射，而不是式 (13)

令 $\Phi(U)=|U|^2$。对任意实对角矩阵 $D_L,D_R$，切向扰动

\[
dU=iD_LU+iUD_R
\]

只改变行相位和列相位，故

\[
dH=2\operatorname{Re}(\bar U\odot dU)=0.
\]

除去重复的全局相位，这至少给出 $2N-1$ 个不可见方向。另一方面，$dH$
必须满足零行和与零列和，其目标切空间维数是 $(N-1)^2$。因此

\[
\operatorname{rank}d\Phi_U\le N^2-(2N-1)=(N-1)^2.
\]

在 $U=I$ 或置换矩阵处退化更严重。若 $dU=K$ 且 $K^\dagger=-K$，
对角元素的一阶模方变化为零，非对角位置又有 $U_{ij}=0$，所以
$d|U|^2=0$，整个 Jacobian 秩为零。当前 float64 诊断在一个固定 seed 的
$N=4$ 非退化点达到上界 9，而在 identity 点返回 0。前者是局部数值证据，
不应被表述成对所有 $N$ 和所有 $U$ 的满秩定理。训练上必须监控 Jacobian
奇异值或 off-diagonal mass，并避免在没有破对称机制时从精确 identity 学习 $H$。

### 定理 3.4：实环唯一固定点

假设 `σ` 在 `L∞` 下 1-Lipschitz（恒等、ReLU、tanh 均满足）。定义

\[
T(h)=\sigma(qH_\beta h+\alpha J).
\]

对任意 `h₁,h₂`，

\[
\|T(h_1)-T(h_2)\|_\infty
\le q\|H_\beta(h_1-h_2)\|_\infty
\le q\|h_1-h_2\|_\infty.
\]

只要 `0<α≤1`，就有 `0≤q<1`。Banach 不动点定理因此给出唯一 `h*`，并且

\[
\|h_K-h^*\|_\infty\le q^K\|h_0-h^*\|_\infty.
\]

若 `σ(0)=0` 且 `h₀=0`，还可得 `||h*||∞≤||J||∞`，故误差不超过 `q^K||J||∞`。

在线时若从旧状态 warm-start，同一新输入下两个初态的影响至多按 `q^K` 衰减。当前常用 `α=0.3,K=12` 时 `q^K≈0.01384`，也就是说“充分松弛”几乎抹掉上一时刻状态。若要把环当作短时记忆，应该使用 `K=1~3` 或单独的小 `α` 慢环，而不是同时声称“每个 token 到平衡”和“长期保留激活记忆”。

### 定理 3.4a：时变环具有统一收缩，但通常没有共同固定点

令

\[
T_t(h)=\sigma(qH_t h+\alpha J_t),\qquad q=1-\alpha,
\]

其中每个 $H_t$ 都双随机，$\sigma$ 在 $\ell_\infty$ 下 1-Lipschitz。
若两条轨迹经历同一 $H_t,J_t$ 序列，则逐步应用收缩不等式得到

\[
\|h_t-\tilde h_t\|_\infty\le q^t\|h_0-\tilde h_0\|_\infty.
\]

该界与 $H_t$ 的具体取值无关，因此允许慢时间尺度在线更新 $U_t$。但是它只
约束两条受同一算子序列驱动的轨迹，不能推出所有 $T_t$ 共享一个固定点。若
$h_t^*=T_t(h_t^*)$ 是瞬时固定点，真实一步状态 $h_t=T_t(h_{t-1})$ 满足

\[
\|h_t-h_t^*\|_\infty
\le q\|h_{t-1}-h_{t-1}^*\|_\infty
+q\|h_t^*-h_{t-1}^*\|_\infty.
\]

所以在线 sidecar 还需要对参数变化施加 trust region，并独立监控旧域 probe；
酉性和统一收缩都不能保证冻结主干的任务性能不下降。

### 命题 3.4b：有符号时间状态的逐步上界

对第 (j) 个时间环，写成

\[
h_{j,t}=\sigma((1-\lambda_j)h_{j,t-1}H_{j,t}^{\mathsf T}+b_{j,t}),
\]

其中 (b_{j,t}) 合并当前写入与 promotion。若 σ 为 1-Lipschitz 且
(\sigma(0)=0)，双随机 (H_{j,t}) 满足

\[
\boxed{
\|h_{j,t}\|_\infty
\le(1-\lambda_j)\|h_{j,t-1}\|_\infty+\|b_{j,t}\|_\infty}.
\]

证明只需使用
(\|\sigma(z)\|_\infty\le\|z\|_\infty) 和
(\|hH^{\mathsf T}\|_\infty\le\|h\|_\infty)。它对 signed state 成立，
不依赖非负锥或 (\ell_1) 质量守恒。当
(\|b_{j,t}\|_\infty\le B_j) 时，迭代该不等式还给出渐近上界
(B_j/\lambda_j)。实现使用浮点矩阵的实测诱导范数，同时返回
解析界、实测范数与数值 violation。

### 命题 3.4c：有限固定点求解的 residual 证书

对任意收缩模量不超过 $q$ 的 $T$，设返回状态为 $\hat h$，绝对 residual
为 $r=\|T(\hat h)-\hat h\|_\infty$。由三角不等式，

\[
\|\hat h-h^*\|_\infty
\le r+q\|\hat h-h^*\|_\infty,
\]

因此

\[
\boxed{\|\hat h-h^*\|_\infty\le r/\alpha}.
\]

伴随映射在 $\|D\|_\infty\le1$ 时同样是 $q$-收缩，故其绝对 residual
也给出 `residual/alpha` 误差界。代码用相对 residual 作尺度化停止判据，但证书
始终由绝对 residual 计算。只要用户请求认证，正向或伴随任一未收敛，事务默认
整笔拒绝，不更新参数、OGD 记忆或外部 `grad_x`。

### 推论 3.5：线性固定点闭式解

`σ` 为恒等映射时，

\[
h^*=\alpha(I-qH_\beta)^{-1}J.
\]

因为 `ρ(qH_β)≤q<1`，逆存在，并有 Neumann 展开

\[
\alpha(I-qH_\beta)^{-1}
=\alpha\sum_{t=0}^{\infty}q^tH_\beta^t,
\]

其诱导无穷范数不超过 1。这说明固定点映射稳定，但 `α` 越小，迭代越慢，且关于 `H` 的灵敏度可按 `1/α` 增大。

### 定理 3.6：复数酉环收敛

复数模式下 `||Uv||₂=||v||₂`，所以

\[
\|qUh_1+\alpha J-(qUh_2+\alpha J)\|_2=q\|h_1-h_2\|_2.
\]

因此同样有唯一固定点。后续 `abs` 测量引入非线性，但不影响内部线性固定点的存在。

## 4. 精确的 BPTT-free 隐式梯度

“不展开 K 步”并不意味着梯度必须近似。若正向与伴随固定点求解收敛，隐函数微分可以给出精确梯度；有限步数才带来几何衰减的求解误差。

令

\[
D=\operatorname{diag}(\sigma'(qH_\beta h^*+\alpha J)),
\qquad g=\nabla_{h^*}L.
\]

固定点微分满足

\[
(I-qDH_\beta)\,dh=D(q\,dH_\beta h^*+\alpha\,dJ).
\]

定义传统伴随 `λ`：

\[
(I-qH_\beta^\top D)\lambda=g.
\]

代码为了让注入梯度更简洁，迭代的是 `p=αλ`：

\[
p=qH_\beta^\top Dp+\alpha g.
\]

因此非线性闭式解是

\[
\boxed{p=\alpha(I-qH_\beta^\top D)^{-1}g},
\]

而不是把线性公式中的 $D$ 省略。等价的行向量写法为
$p_{\rm row}=\alpha g_{\rm row}(I-qDH_\beta)^{-1}$。只有 $D=I$ 时才退化为
论文旧版的线性 resolvent。微分式中的 $q h^*dH^\top$ 是一阶项；伴随法消去
$dh^*$ 后仍通过下面的外积完整保留它，并未把它作为二阶项截断。

于是严格链式法则为

\[
\boxed{\nabla_{H_\beta}L=\frac{q}{\alpha}(Dp)(h^*)^\top},
\qquad
\boxed{\nabla_JL=Dp}.
\]

对 batch 求和即可；若 `g` 已来自 mean-reduced loss，就不能再除一次 batch size。旧代码漏掉 `q/α` 并重复除以 batch size，虽不总改变方向，却错误改变了不同 `α` 和 batch size 下的学习尺度。

因为 $\|H_\beta^\top D\|_\infty\le1$，Neumann 级数给出

\[
\|(I-qH_\beta^\top D)^{-1}\|_\infty\le\frac1\alpha,
\qquad \|p\|_\infty\le\|g\|_\infty.
\]

所以常规伴随 $\lambda=p/\alpha$ 的上界会随 $1/\alpha$ 增长，缩放伴随 $p$
本身不必爆炸。然而 transition gradient 显含 $q/\alpha$，求解误差证书也含
$1/\alpha$；小 $\alpha$ 仍会显著增加迭代数和梯度灵敏度，不能只靠酉性误差
判断数值安全。

对于 `H_β=(1-β)I+βH`，

\[
\nabla_HL=\beta\nabla_{H_\beta}L,
\qquad
\frac{\partial L}{\partial\beta}
=\langle\nabla_{H_\beta}L,H-I\rangle_F.
\]

### 定理 4.1：`H=|U|²` 的实损失梯度

采用实 Frobenius 内积 `⟨X,Y⟩=Re tr(X†Y)`。因为

\[
dH=2\operatorname{Re}(\bar U\odot dU),
\]

所以

\[
\boxed{\nabla_UL=2\,\nabla_HL\odot U}.
\]

旧公式中的 `∇H⊙|U|²` 不是这个导数；`|U|²` 还显式依赖 `Ū`，所以称该路径为 “holomorphic” 也不准确。

### 定理 4.2：Cayley 微分与坐标拉回

令 `R=(I+A)^{-1}`。对

\[
U=(I-A)(I+A)^{-1}
\]

求微分，乘积法则给出精确恒等式（不是“小扰动近似”）：

\[
dU=-2R(dA)R.
\]

于是环境空间梯度拉回为

\[
G_A^{amb}=-2R^\dagger(\nabla_UL)R^\dagger.
\]

限制 `dA†=-dA` 后，合法坐标梯度是

\[
\boxed{\nabla_AL=\operatorname{skew}(G_A^{amb})
=\tfrac12(G_A^{amb}-(G_A^{amb})^\dagger)}.
\]

代码对 `A_real,A_imag` 做 `-η∇A` 更新。由于每次前向都重新投影成反 Hermitian `A`，任意步长下生成的 `U` 仍酉；而对足够小的 `η`，Taylor 展开给出

\[
L(A-\eta\nabla_AL)
=L(A)-\eta\|\nabla_AL\|_F^2+O(\eta^2),
\]

所以非驻点处存在严格下降步长。这里同时获得了“约束不被破坏”和“局部下降”两个不同保证。

### 有限求解误差

线性情形中，从零初始化的正向误差与伴随误差分别至多按 `q^K`、`q^M` 衰减。外积不等式

\[
\|p_Mh_K^\top-p^*(h^*)^\top\|_F
\le\|p_M-p^*\|_2\|h_K\|_2
+\|p^*\|_2\|h_K-h^*\|_2
\]

因此梯度误差为 `O(q^K+q^M)`。正向和伴随 residual 分别认证状态误差，但完整
非线性梯度还包含 $D(\hat h)-D(h^*)$；要把状态证书转换成统一的梯度误差界，
还需要 `σ'` 局部 Lipschitz、状态/伴随有界以及 transition-gradient 尺度。
ReLU 的不可微点使用次梯度，不能声称处处经典可微。当前实现因此同时报告
solver 证书和直接/autograd 梯度误差，而不把 `converged=True` 等同于任意精度的
完整梯度证书。

## 5. 表达能力：线性环严格等价于 LoRA

### 定理 5.1：单个线性环的秩上界

设外挂环把冻结模型特征 `x∈R^d` 修正为

\[
\Delta y=W_o h^*,\qquad J=W_{up}W_{down}x,
\]

且内部无非线性。代入闭式固定点：

\[
\Delta y=
\underbrace{W_o\alpha(I-qH_\beta)^{-1}W_{up}}_{B_{eff}}
\underbrace{W_{down}}_{A_{eff}}x.
\]

因此 `rank(ΔW)≤r`。这就是一个普通 LoRA；环的 resolvent 可完全吸收到上投影中。多个并行线性环相加，秩最多为各环 `r` 之和，也等价于一个更高秩 LoRA。

所以线性版本的合理价值是参数化、稳定性与在线更新规则，而不是更高的函数表达力。原分析中“上限是核方法”的说法既不精确也没有必要；这里有更强的有限维线性等价结论。

### 非线性环何时可能更强

当注入或状态使用 ReLU/tanh/GELU，或者路由器依赖上下文时，resolvent 不能合并成固定矩阵，环成为隐式非线性适配器。此时多环可以形成 mixture-of-experts，但代价是：

- 每个 token 需要迭代，不能合并进基座权重；
- 多环独立收敛不等于整个 Transformer 残差映射收敛；
- 路由漂移与共享参数仍会造成遗忘；
- 表达优势必须和同参数 MLP/LoRA/adapter 实验比较。

### 参数不可辨识与近单位初始化

对任意对角酉矩阵 `D₁,D₂`，`|D₁UD₂|²=|U|²`。因此 `H` 看不到至少 `2N-1` 个相位规范自由度。更严重的是，在 `U=I` 或任意置换矩阵处，`U(ε)=I+εK+O(ε²)` 且 `K†=-K`，逐元素模方的一阶变化全部为零，所以

\[
d|U|^2\big|_{U=I}=0.
\]

当前 `A` 的 `0.01` 小随机初始化接近单位阵，unistochastic 路径的连接梯度天然偏小。可以比较 Haar/块旋转初始化、冻结随机 `H`、或先只训练注入/读出再慢速解冻 `U`。

## 6. “正交更新”必须区分三种概念

1. **状态转移酉/正交**：`U†U=I`，保证传播等距；不保证旧知识不被覆盖。
2. **参数更新位于合法切空间**：保证更新后仍满足结构约束；不保证旧任务损失不变。
3. **新梯度与旧知识梯度子空间正交**：这才给出一阶抗遗忘结论。

### 定理 6.1：异构学习率下的正交梯度一阶保证

当前模型对读出、注入、Cayley 坐标和目标态使用不同学习率。令每个参数块的实际正步长组成对角矩阵 `D`，历史白化梯度 `D^{1/2}g_j` 的正交基为 `Q`，并定义

\[
w=D^{1/2}g,\qquad
w_\perp=(I-QQ^\top)w,\qquad
\boxed{\Delta\theta=-D^{1/2}w_\perp}.
\]

若受保护旧损失满足 `D^{1/2}g_old∈span(Q)`，则

\[
g_{old}^\top\Delta\theta
=-(D^{1/2}g_{old})^\top w_\perp=0.
\]

同时当前损失的一阶变化恰为

\[
g^\top\Delta\theta
=-(D^{1/2}g)^\top w_\perp
=-\|w_\perp\|_2^2\le0.
\]

标量学习率 `D=ηI` 时退化为普通 OGD。若旧损失梯度为 `L`-Lipschitz，下降引理只给出

\[
L_{old}(\theta+\Delta\theta)-L_{old}(\theta)
\le \frac{L}{2}\|\Delta\theta\|_2^2.
\]

因此这是单步、局部、一阶保证，不是任意多步或分布漂移下的零遗忘定理。参数移动后旧梯度会陈旧；低秩截断遗漏方向；当 `w⊥≈0` 时还会出现“稳定但学不动”的可塑性耗尽。实现要求记忆有效期内参数布局与**相对**学习率不变，并用两遍 modified Gram–Schmidt 控制数值误差。

### 推论 6.2：分块 OGD 与 trust-region 缩放不破坏整体正交性

把外挂参数拆为 MQR、LoRA 等块。第 `b` 块独立使用历史基 `Q_b`，并允许原子 trust region 给出任意标量 `c_b∈[0,1]`：

\[
\Delta\theta_b=-c_bD_b^{1/2}(I-Q_bQ_b^\top)D_b^{1/2}g_b.
\]

若每个受保护旧梯度块均满足 `D_b^{1/2}g_{old,b}∈span(Q_b)`，则每一块的内积分别为零，故

\[
g_{old}^\top\Delta\theta
=\sum_b g_{old,b}^\top\Delta\theta_b=0.
\]

同时当前损失的一阶变化为各块非正项之和：

\[
g^\top\Delta\theta
=-\sum_b c_b\left\|(I-Q_bQ_b^\top)D_b^{1/2}g_b\right\|_2^2\le0.
\]

因此 MQR 完整向量与外部 LoRA 分别投影、分别裁剪，在记忆采样同步且各旧梯度块确实被保存时，仍有整体一阶保证。这种分块约束比单个全局投影更强，可能牺牲更多可塑性。`analysis/mqr_proof_verify.py` 已数值验证不同块使用不同缩放时的等式。

`OrthogonalGradientMemory` 当前在完整实参数坐标上存稠密基。Cayley 的实/虚原始坐标会自动投影成反 Hermitian `A`，所以更新后仍保酉；但该欧氏 OGD 度量依赖坐标选择。2B 场景必须改用分块基、随机 sketch 或功能/Jacobian 子空间，不能照搬 `m×P` 稠密存储。

### 推论 6.2a：慢参数屏蔽与 OGD 可同时严格满足

直接对普通 OGD 结果做坐标 mask 会重新引入历史梯度分量，
因此不能保证一阶正交。令 (M=M^\top=M^2) 为当前 tick 允许
更新的坐标投影，(w=D^{1/2}g)，则在 `range(M)` 内同时满足
历史正交的最近向量为

\[
z=Mw-MQ^\top(QMQ^\top)^+QMw.
\]

由 (\operatorname{range}(QM)=\operatorname{range}(QMQ^\top)) 可得
(Qz=0)；由 (z\in\operatorname{range}(M)) 可得被屏蔽坐标严格为零。
该式是正交投影，所以 (w^\top z=\|z\|_2^2)，更新
(\Delta\theta=-D^{1/2}z) 仍满足

\[
g_{old}^\top\Delta\theta=0,
\qquad
g^\top\Delta\theta=-\|z\|_2^2\le0.
\]

代码用小型 (r_{\rm OGD}\times r_{\rm OGD}) Gram 矩阵的伪逆实现该受限
投影。慢 Cayley 坐标因此可在非调度 tick 保持精确不变，而快参数
仍使用同一完整布局 OGD 记忆。`retained_norm` 低于阈值只产生
可塑性告警，不会强制违反保护子空间的更新。

### 定理 6.3：增量环隔离的精确无干扰条件

令总残差为

\[
R(x)=\sum_k g_k(x)R_k(x).
\]

若旧环与旧路由完全冻结，并且对所有受保护旧输入，新环门值严格为零，那么新增环后模型输出完全不变。当前 `OnlineMultiRingClassifier` 的显式上下文 Top-1 路由是这个定理的特例：只更新被选环；容量耗尽默认报错而不静默复用。因此 Digits 实验中学习 B 时 A 环参数和无状态 logits 的漂移都精确为零。实验性余弦新颖性路由不具备这个保证；误路由、共享主干更新或 `least_used` 溢出复用都会使结论失效。

推荐把两者结合：旧环冻结提供任务级隔离，当前活动环内部使用低秩 OGD 提供样本级一阶保护。

### 定理 6.4：受控写门不破坏收缩，并给出干扰响应上界

Temporal MQR 第 `j` 个状态现在允许外部写门
$g_{j,t}\in[0,1]$：

\[
h_{j,t}=\sigma\!\left(q_jh_{j,t-1}H_j^\top
+\kappa_jg_{j,t}J_j(x_t)\right),\qquad q_j=1-\lambda_j<1.
\]

设 `σ` 为 1-Lipschitz，`H_j` 双随机。对相同输入和门控、不同初态，

\[
\|h_{j,t}-h'_{j,t}\|_\infty
\le q_j\|h_{j,t-1}-h'_{j,t-1}\|_\infty.
\]

证明只需使用
$\|(h-h')H_j^\top\|_\infty\le\|h-h'\|_\infty$；注入项完全
相消。因此写门不会改变唯一受迫轨迹和指数遗忘保证。

进一步，对两组输入/门序列，记
$e_t=\|g_tJ(x_t)-\tilde g_tJ(\tilde x_t)\|_\infty$，递推可得严格
扰动界

\[
\boxed{
\|h_t-\tilde h_t\|_\infty
\le q^t\|h_0-\tilde h_0\|_\infty
+\kappa\sum_{s=1}^{t}q^{t-s}e_s.}
\]

若干扰帧的门严格为零，其直接贡献精确为零，只剩目标 cue 自身按
$q^t$ 衰减；无门条件则累积全部干扰卷积。若
$\|J(x_t)\|_\infty\le M$，还有

\[
\|h_t\|_\infty\le q^t\|h_0\|_\infty
+\kappa M\sum_{s=1}^{t}q^{t-s}g_s
\le q^t\|h_0\|_\infty+\frac{\kappa M}{\lambda}.
\]

v3 的门是外部控制信号；oracle 收益只能证明选择性写入的价值。v4 另加了
下面的辅助监督控制器，但它仍不是从延迟任务奖励中学到的策略。

### 定理 6.4a：辅助事件门的当前步零泄漏与梯度分离

令门控制器参数为 $\phi$、MQR/读出参数为 $\theta$。当前事件特征为 $e_t$，
控制器定义

\[
a_t=f_{\phi_t}(e_t),\qquad
p_t=\operatorname{sigmoid}(a_t/\tau),\qquad
g_t=m+(1-m)p_t.
\]

实现的当前步不是端到端门控，而是

\[
(\hat y_t,h_t)=F_{\theta_t}\!\left(
x_t,h_{t-1},\operatorname{stopgrad}(g_t)\right).
\]

只有上述量都确定并完成当前写入/预测后，辅助事件标签 $c_t$ 才用于

\[
\ell_g(a_t,c_t)
=-\frac1N\sum_i\left[
w_+c_i\log p_i+(1-c_i)\log(1-p_i)\right].
\]

因此，对任意两个仅 $c_t$ 不同、而
$(x_t,e_t,h_{t-1},\theta_t,\phi_t)$ 完全相同的执行，当前
$(g_t,h_t,\hat y_t)$ 逐位相同。形式上

\[
\frac{\partial(g_t,h_t,\hat y_t)}{\partial c_t}=0,\qquad
\nabla_\phi\ell_{\mathrm{task},t}=0.
\]

第一个等式来自 $c_t$ 不出现在当前前向映射中；第二个等式来自梯度截断。
调用结束后两次执行的 $\phi_{t+1}$ 可以不同，这只影响未来步。
外部门存在时，它覆盖 $g_t$ 对状态的作用；控制器仍可用同一个
$(e_t,c_t)$ 在后台学习，故其梯度与外部门、MQR 状态和任务标签无关。

令 $z_i=a_i/\tau$，带正类权重 BCE 的逐 logit 导数严格为

\[
\frac{\partial\ell_g}{\partial a_i}
=\frac{w_+c_i(p_i-1)+(1-c_i)p_i}{N\tau}.
\]

损失作用于基础 Bernoulli 概率 $p$，不是带下限的实际门 $g$。这避免梯度再
乘 $(1-m)$，但若 $m>0$，所谓“关闭”仍有最多由定理 6.4 扰动界刻画的残余
注入。

门参数使用独立完整向量 OGD。令
$w_g=D_g^{1/2}\nabla_\phi\ell_g$，历史白化基为 $Q_g$，则

\[
\Delta\phi=-c_gD_g^{1/2}(I-Q_g^\top Q_g)w_g,\qquad c_g\in[0,1],
\]

其中本实现的基向量按行存储。于是当前门损失的一阶变化为

\[
\nabla_\phi\ell_g^\top\Delta\phi
=-c_g\|(I-Q_g^\top Q_g)w_g\|_2^2\le0,
\]

且对基中保存的旧门梯度一阶内积为零。任务块与门块直到事务尾部才同时提交；
任一块有更新时全局参数版本只增加一次。该结论仍要求固定参数布局/相对步长，
也不保证非凸控制器的全局收敛或长期零遗忘。

### 命题 6.4b：本次 marker 门任务在函数类中可实现

实验事件特征只有 $e\in\{0,1\}$。即使 rank 为 1，取
$f_\phi(e)=v\tanh(we)+b$ 且 $w\ne0$，因为
$\tanh(0)=0$、$\tanh(w)\ne0$，对任意期望 logits $(a_0,a_1)$ 可取

\[
b=a_0,\qquad v=\frac{a_1-a_0}{\tanh(w)}.
\]

所以 cue/non-cue 映射确实包含在低秩门函数类中。该表示性证明不等于 SGD
必然找到它；本次 3-seed 数值实验只提供有限样本下的优化证据。若没有 marker、
即时 $c_t$，或只给稀疏远端奖励，本命题和定理 6.4a 都不能解决信用分配。

### 定理 6.4c：未来收益影子归因的因果语义与最优门

v5 不再构造事件标签。令决策前特征
$z_t=Z(x_{\le t},y_{<t},h_{t-1})$，因此未来反馈 $y_{\ge t}$ 不在
$z_t$ 的计算图中。从同一个 $h_{t-1}$ 建立两个状态分支，除当前慢环动作
$a_t\in\{0,1\}$ 外完全相同：

\[
h_t^{(a)}=\sigma\!\left(qh_{t-1}H^\top
+aJ(x_t)+r_t\right),\qquad a\in\{0,1\}.
\]

从 $t+1$ 到反馈时刻 $\tau$，两个分支使用实际执行轨迹中相同的输入、门和
晋升项。定义有限时域单次干预收益

\[
A_{t\to\tau}=\ell_\tau(h_\tau^{(0)})
-\ell_\tau(h_\tau^{(1)})-c_{mem}.
\]

因此 $A_{t\to\tau}$ 是“固定未来动作序列”条件下，当前写入的配对个体处理
效应：除了 $a_t$ 外没有分支噪声。它不是让未来策略也随反事实状态变化的
总策略效应；若门会强烈依赖被改变的状态，这两个 estimand 可能不同。

门输出 $v_\phi(z)$，以
$\widetilde A=\operatorname{clip}(A/s,-C,C)$ 为目标最小化

\[
R(v)=\mathbb E[(v(z)-\widetilde A)^2].
\]

在 $\widetilde A$ 平方可积、并允许任意可测 $v$ 时，用条件期望的正交分解

\[
R(v)=\mathbb E[\operatorname{Var}(\widetilde A\mid z)]
+\mathbb E[(v(z)-\mathbb E[\widetilde A\mid z])^2]
\]

立即得到几乎处处唯一的风险最优解

\[
\boxed{v^*(z)=\mathbb E[\widetilde A\mid z].}
\]

若不裁剪、没有共享容量约束且两动作差值就是 $A$，则
$a^*(z)=\mathbf1[v^*(z)>0]$ 最小化给定 $z$ 的期望未来损失加记忆成本。
当前低秩函数类、有限在线数据、目标裁剪和优化误差意味着实现只能逼近该规则。
门更新使用签发时缓存的 $z_t$ 在当前 $\phi$ 下重算，所以它是**特征回放梯度**；
签发版本差被显式报告。未来标签只在影子状态和当前动作均确定后进入损失，故
改变反馈标签不能改变签发时特征、动作、真实状态或 logits。

### 定理 6.4d：晋升不改变收缩；不可预测相关性的下界

v5 允许外部晋升 $r_t$：

\[
h_t=\sigma(qh_{t-1}H^\top+g_tJ(x_t)+r_t).
\]

对相同 $(x_t,g_t,r_t)$ 的两个轨迹，受迫项作差消失；完全沿用定理 6.4 的
证明可得

\[
\|h_t-h'_t\|_\infty\le q\|h_{t-1}-h'_{t-1}\|_\infty.
\]

所以候选延期晋升不改变齐次 Jacobian、Cayley/unistochastic 约束或指数遗忘率。
若 `tanh` 为状态激活，状态逐坐标有界；恒等激活时若
$\|g_tJ(x_t)+r_t\|_\infty\le B$，递推还给出

\[
\|h_t\|_\infty\le q^t\|h_0\|_\infty+\frac{B}{1-q}.
\]

但“延迟监督”不产生先知能力。令未来有用性为二值变量 $Y$，并假设
$0<P(Y=1)<1$。若决策时
$P(z\mid Y=1)=P(z\mid Y=0)$，则 $Y$ 与 $z$ 独立，因而

\[
P(Y=1\mid z)=P(Y=1).
\]

所以在预测二值有用性的 0-1 风险下，任何只依赖当前 $z$ 的确定性或随机门
都不能优于最优常数先验分类器。若进一步有完整收益
$A\perp z$（仅有 $Y\perp z$ 不足以排除收益幅度携带信息），才可推出
$\mathbb E[A\mid z]=\mathbb E[A]$，此时期望收益写门也不能优于最优常数动作。
缓存可在
反馈后补写，并让后续具有相同统计结构的事件受益，却不能修复已经完成的第一次
不可预测查询。v5 Digits 特意让目标在位置上随机、但在新颖性上可识别；它验证
学习可观测统计规律，不推翻这个下界。

### 定理 6.4e：过去波动路由的因果性、专家零梯度与 OGD 隔离

v6 在不输入任务 ID 的前提下，为收益门增加一个只用于路由的
过去上下文波动迹。令观测从 $t=1$ 开始，并定义

\[
d_t=\frac12\left(1-\frac{x_t^\top x_{t-1}}
{\|x_t\|_2\|x_{t-1}\|_2}\right)\in[0,1],
\]

零向量情形按实现取余弦相似度为 0，且 $d_1=0$。代码的精确启动条件是

\[
v_1=0,\qquad
v_{t+1}=\begin{cases}
d_t,&t\le2,\\
\rho v_t+(1-\rho)d_t,&t\ge3,
\end{cases}
\qquad 0\le\rho<1.
\]

第一个有效差分直接初始化迹，之后才进入 EMA。在时刻 $t$ 做决策时，
运行时只读取已于上一个事务末尾提交的 $v_t$，并以

\[
e_t=\mathbf1[v_t>\tau]
\]

硬路由两个收益专家。因此当前 $d_t$ 与当前/未来标签都不可能改变
$e_t$。需要区分：普通特征 `context_change` 可在专家内读取当前
$d_t$；上述严格过去性特指分专家路由，不是声称整个门忽略当前输入。

将专家参数写为 $\theta=(\theta_0,\theta_1)$，硬路由输出为

\[
G(z;\theta)=g_{e(z)}(z;\theta_{e(z)}).
\]

对任意仅依赖 $G$ 的可微损失 $L$，链式法则立即给出

\[
\nabla_{\theta_{1-e}}L=0.
\]

所以 SGD 或任意不混合坐标块的预条件更新都不会改写未选专家。
当前 OGD 也保留该性质，但需要把条件说完整。令白化参数空间正交分解为
$V_0\oplus V_1$。运行时每次只处理一个候选，因此每个原始白化梯度
$w_t$ 只支撑在 $V_{e_t}$。修正 Gram–Schmidt 对互相正交的两个块做正交化
不会创造另一块分量，故历史基可写成 $Q=Q_0\oplus Q_1$，投影算子为

\[
I-Q^\top Q=(I-Q_0^\top Q_0)\oplus(I-Q_1^\top Q_1).
\]

于是投影后梯度仍只在被选块，全向量范数裁剪又只乘一个标量，
未选专家的更新精确为零。`test_utility_mqr.py` 已对“先记忆专家 0
梯度，再对专家 1 做 OGD”的路径逐位验证专家 0 零漂移。

路由只改变慢环方程的外部 forcing $a_tJ(x_t)$。对相同已实现路由/写入
序列的两个轨迹，注入作差消失，所以定理 6.4 的 $q$-收缩原样成立。
对不同路由序列，不能说状态相同，只能使用定理 6.4 的受迫扰动卷积上界；
齐次 Jacobian、Cayley 保酉性和 unistochastic 传播仍未改变。

精确隔离会被以下任一情形破坏：软路由；把不同专家样本混成一个梯度并
将其作为 OGD 基；专家间共享参数、动量或权重衰减；共享 core/readout 同时
更新；或上下文误路由。因此该定理证明的是“正确路由后的专家参数隔离”，
不是无条件零遗忘。

### 定理 6.5：有界轨迹反馈的精确信用路径及其单边界

对同一参数版本签发并连续提交的票据，令

\[
h_k=F_\theta(h_{k-1},x_k,w_k),\qquad
L_{s:t}=\frac{1}{S}\sum_{k=s}^t a_k\ell_k(h_k),
\quad S=\sum_{k=s}^t a_k>0.
\]

若在反馈前没有参数更新，按已提交的输入、写门和初态重放该有限轨迹，再对
\(L_{s:t}\) 自动微分，得到的就是这段有限计算图的精确 BPTT 梯度：

\[
\nabla_\theta L_{s:t}
=\frac1S\sum_{k=s}^t a_k
\left(\frac{\partial h_k}{\partial\theta}\right)^\top
\nabla_{h_k}\ell_k.
\]

它不同于单票据更新：单票据只微分一个已提交转移，因而截断其后的任务信用；
有界轨迹反馈允许后续 query loss 更新较早的注入、读出与转移参数。实现限制
轨迹长度、要求同一 stream/参数版本，并在一次原子事务内提交，所以该结论不延伸
到无限历史或跨版本异步票据。

若第 \(j\) 个环的状态激活导数满足 \(\|D_k^{(j)}\|_2\le1\)，则

\[
\left\|\frac{\partial h_t^{(j)}}{\partial h_s^{(j)}}\right\|_2
\le (1-\lambda_j)^{t-s}.
\]

这是未来信用的**上界**，不是正下界。激活饱和、读出零空间、损失梯度为零、
模平方 Jacobian 奇异或相互抵消都可使实际梯度精确为零。因此数值上观察到
`earliest_input_future_gradient>0` 只能证明计算图连通，不能证明信用足够大、方向
有益或 MQR 优于其他递归核心。

### 定理 6.5a：延迟票据精确重建签发时读出梯度

令票据签发时缓存状态向量 $z_t$ 和旧策略 logits
$a_t=W_tz_t+b_t$。对 batch 平均交叉熵，令
$P_t=\operatorname{softmax}(a_t)$、标签矩阵为 $Y_t$，则

\[
\nabla_W\ell_t=\frac{1}{B}(P_t-Y_t)^\top z_t,
\qquad
\nabla_b\ell_t=\frac{1}{B}\mathbf 1^\top(P_t-Y_t).
\]

所以只缓存 `(z_t,a_t)` 就能在反馈到达后精确重建**签发版本**的读出梯度，
无需保存 autograd 图、旧递归状态或完整参数快照。实现只允许票据消费一次，
并保存签发参数版本 `v_t`。若消费前已有其他更新，梯度在当前版本上是异步
陈旧梯度，不再保证当前损失下降。若签发样本损失的梯度为 `L`-Lipschitz，

\[
\|\nabla\ell_t(W_v)-\nabla\ell_t(W_t)\|
\le L\|W_v-W_t\|,
\]

这说明版本差和累计更新范数必须作为运行指标。对 softmax 线性读出，固定
特征下可取一个保守的 batch 光滑度界
$L\le\frac{1}{2B}\sum_i\|z_i\|_2^2$。若要延迟更新注入、Cayley 坐标或
LM LoRA，仅缓存读出特征不够，必须保存参数/特征快照或可复现重算；当前实现
明确拒绝伪造这种全结构梯度。

### 命题 6.6：keyed state bank 的隔离边界

把运行时状态表示为映射
$S:\texttt{stream\_id}\mapsto(h_1,\ldots,h_m)$。一次针对键 `k` 的推理
事务只把 `S[k]` 替换为其下一状态，因此对任意 `k'≠k`，有
$S_{after}[k']=S_{before}[k']$。这给出交错会话的**激活状态零串扰**；它
不保证共享参数零串扰，因为任一监督更新仍会修改所有会话共享的读出/注入。
容量默认 `error`，选择 LRU 时被淘汰键的连续记忆保证终止，且事务返回确切
淘汰 ID。state bank、访问次序、使用计数、待反馈票据和两个 OGD 基均进入
checkpoint。门控制器加入后还保存第三个、与同步任务及延迟读出均分离的门
OGD 基，以及门更新/标签计数。

## 7. 面向 2B LLM 的可实施架构

当前仓库已有一个 **1B 级机制桥接**：`mqr/minicpm.py` 从本地 MiniCPM5 AWQ checkpoint 构造冻结 FP16 隐状态编码器，在末层 `q_proj/v_proj` 加 LoRA，并接受 MQR 返回的外部特征梯度；`experiments/minicpm_go_online.py` 已用 26 类围棋动作头闭合前向和更新。Temporal sidecar 已补上有界 token/会话状态库、读出级延迟反馈队列，以及小状态上的未来收益影子归因原型，但这还不是 2B 生成式集成：尚未经过冻结 LM head 的 next-token loss，也未把 utility ticket 与 KV-cache 生命周期对齐；逐 token 复制 LM/KV 影子分支在资源上不可行。

该桥接的链式法则是严格的。令 `φ` 为 LoRA，`ψ` 为 MQR，`z(φ)=LN(f_{θ₀,φ}(p))`。隐式伴随给出更新前参数处的 `g_z=∇_z L(φ_t,ψ_t)`，随后向主干特征图执行 VJP：

\[
\nabla_\phi L=J_{z,\phi}^{\top}g_z.
\]

虽然工程按“提交 MQR、再提交 LoRA”执行，LoRA 使用缓存的更新前 `g_z` 与原特征图，故两个梯度都来自同一 `(φ_t,ψ_t)`，等价于一阶 Jacobi 更新。完整实现和实验证据见 `analysis/minicpm_sayuri_online_report.md`。

安全提交使用候选事务：先快照 sidecar 参数与 OGD 基，应用已裁剪候选
更新，再在固定 probe 上计算原生 logits KL、输出/fast-state 漂移和
(H) 漂移。任一预登记界被突破时，参数、OGD 基、参数版本和外部
`grad_features` 授权一起回滚。统一 Agent 可审计签发样本的局部变化；
MiniCPM `constancy_closure` 则允许调用者使用旧域 prompt 的冻结 LM logits。
该 LoRA-only 接口只能认证 policy KL 与输出漂移；recurrent-state 和
Cayley-transition 预算属于统一 Agent，误传给 LoRA 接口会显式拒绝，避免形成
“已检查”但实际未观测的安全假象。
这个机制限制单步破坏，不证明新任务获益或长期无遗忘。

最稳妥的第一版不是“包住整个模型 logits”，而是在冻结 2B 主干最后隐藏层、冻结归一化和 tied LM head 之间插入小残差 sidecar：

\[
z_t=\operatorname{stopgrad}(f_{2B}(x_{\le t})),
\]

\[
J_k=Q^{in}_k\phi(P^{in}_k z_t),
\qquad P^{in}_k\in\mathbb{R}^{r\times d},\ Q^{in}_k\in\mathbb{R}^{N\times r},
\]

\[
h_{k,t}=\text{Ring}_k(J_k,h_{k,t-1}),
\]

\[
\Delta z_{k,t}=Q^{out}_k\psi(P^{out}_k h_{k,t}),
\qquad P^{out}_k\in\mathbb{R}^{r_o\times N},\ Q^{out}_k\in\mathbb{R}^{d\times r_o},
\]

\[
\hat z_t=z_t+\sum_{k\in\operatorname{TopK(router)}}\gamma_k\Delta z_{k,t},
\qquad \ell_t=E^\top\operatorname{Norm}(\hat z_t).
\]

这样 next-token loss 可以只反传到 sidecar，无需保存 2B 主干的反向图。若把环插入多个内部层，适应能力更强，但梯度仍需穿过后续冻结层，训练显存与时延会明显上升，除非改用局部损失。

### 参数量

推荐的 `minimal` 坐标每个环存储 (N^2) 个 Cayley 实参数；
旧 `projected` 模式为 checkpoint 兼容存储 (2N^2)。双低秩输入/输出的
最小坐标总参数约为

\[
P_{ring}=N^2+r(d+N)+r_o(d+N).
\]

以下按 `d=2048`、`r=r_o`、8 个插入位置、每处 4 个环计算：

| N | r | 单环/位置 | 总外挂参数 | 占 2B |
|---:|---:|---:|---:|---:|
| 64 | 8 | 37,888 | 1.21M | 0.061% |
| 128 | 8 | 51,200 | 1.64M | 0.082% |
| 256 | 16 | 139,264 | 4.46M | 0.223% |

参数上完全可行。若只在最终隐藏层外挂，再除以 8。

### 计算与时延

- 当前 Cayley 是稠密复数求解：每次参数变化后每环 `O(N³)`；必须缓存 `U/H`，不能每 token 重算。
- 每个活动环每 token 的松弛为 `O(KN²)`，投影约 `O(r(d+N)+r_o(d+N))`。
- `N=128,r=8,K=2` 时单个活动环/位置约 6.8 万次乘加；Top-1 路由、8 个位置约 54 万次乘加，算术量相对 2B 主干很小，但未融合的小矩阵 kernel 和 Python 循环可能主导实际延迟。
- 不应令 `N≈d`、所有层放多环、每 token 完整求 Cayley 并做 10~20 次松弛；那会丢掉参数高效方法的工程优势。
- 下一步应实现 block-diagonal/Givens/Householder 结构，或仅慢速更新 `U`、快速更新低秩注入/读出。

### 推荐的快慢双时间尺度

- **快参数**：当前活动环的输入/输出低秩矩阵；每个反馈或小批次更新，带 OGD/sketch、范数裁剪。
- **慢参数**：`U/A`；每 32~256 个样本更新一次，用本文精确 Cayley 梯度，随后刷新缓存。
- **稳定记忆**：旧环冻结；新颖性超过阈值时分配新环。
- **容量控制**：周期性 replay/distillation，把相近线性环合并；非线性环合并前必须做功能蒸馏。
- **路由**：先 Top-1/Top-2，加入负载均衡和旧上下文路由保持损失，避免所有环同时被改写。

## 8. 合理性、不合理性与创新性

### 合理部分

- 结构约束是解析的，不需要每步 Sinkhorn 投影。
- 阻尼把酉/双随机传播变成严格压缩，固定点和伴随都稳定。
- 精确隐式梯度不需要保存全部松弛轨迹，适合小型在线 sidecar。
- 冻结主干 + 小环库可以把快速适应与基座知识分离。
- 多时间尺度、warm state、稀疏环路由适合研究持续适应。

### 不合理或被夸大的部分

- “Möbius” 目前没有实际的 Möbius 拓扑或扭转边界；连接是稠密矩阵，“ring”主要是动力学隐喻。
- unistochastic 模式推理只看 `|U|²`，不能称为量子干涉或量子隧穿；整个系统是经典计算。
- unistochastic 集合存在边界/奇点，不能不加限定地称为全局光滑“流形”。
- `HamiltonianOptimizer` 只是 SGD 包装器，不是哈密顿/黎曼优化器。
- 当前 `U_base` 只是冻结的单位/随机酉矩阵，没有预测环境的训练目标，严格说还不是“世界模型”。
- `H` 双随机不等于 `H` 正交，也不普遍保持信号能量。
- “U 保酉”不推出“在线学习不遗忘”；必须加冻结、路由、回放或梯度投影。
- 线性平衡环可完全合并成 LoRA，不能仅凭结构宣称更强表达力。
- 在每个 token 充分收敛会快速丢失上一状态，和长期工作记忆目标冲突。
- 当前 online 已实现“先预测、后反馈更新”、多环分配、梯度方向巩固、读出级延迟 ticket，以及小环状态上的有限时域反事实信用原型；但 LM 全结构信用、稀疏长期奖励、自监督世界模型、回放蒸馏或可靠的无任务 ID 路由仍不存在。

### 创新强度判断

单独看，Cayley 酉参数化、unistochastic 矩阵、阻尼固定点、隐式伴随、LoRA、正交梯度投影和多适配器路由都不是新概念。更可能成立的创新点是它们的组合：

1. **可严格收敛的 unistochastic 隐式 LoRA sidecar**；
2. **在 Cayley 坐标中做精确 BPTT-free 在线更新**；
3. **环库 + 快慢参数 + 切空间 OGD 的持续学习系统**；
4. **同一模块在 `K=1` 时作为稳定递归记忆、在大 K 时作为平衡适配器的双工作模式**。
5. **因果特征门 + 配对影子环 + 延迟晋升的未来效用写入组合**。
6. **过去上下文硬路由 + 分专家效用学习 + 可证的 OGD 参数隔离组合**。

这些目前是“值得验证的组合创新”，还不是已经证明的新颖性或性能创新。论文或专利层面的 novelty 需要系统检索相邻的 unitary RNN、deep equilibrium、reservoir/fast-weight、continual adapter 和 orthogonal-gradient 文献，并用消融证明每个组合部件不可被简单 LoRA 替代。

## 9. 现有实验证据能说明什么

- 当前七个测试入口共 111 项通过：核心测试 `24/24`、在线测试 `7/7`、
  Temporal 测试 `31/31`、Utility 测试 `11/11`、统一 Agent 测试 `12/12`、
  正式竞争 Go 测试 `4/4` 和 Go/MiniCPM 测试 `22/22`。新增覆盖最小 Cayley 坐标、模平方 Jacobian 退化、
  非线性直接伴随、未认证事务原子拒绝、初始 sidecar 零扰动、10×10 无损编码、
  任意棋盘 ko 配对、四头联合概率归一化、signed-state 界、慢 Cayley
  受限 OGD 和 constancy 回滚。
- `analysis/mqr_proof_verify.py` 验证闭式固定点、初态衰减、线性 LoRA 等价和 OGD 一阶正交。
- `analysis/mqr_sidecar_safety_verify.py` 的 20 个 signed-state 扫描和 40 个
  Cayley 漂移扫描全部通过。状态界最大 violation 为 0；实测
  (\|\Delta U\|_F/(2\|\Delta A\|_F)) 不超过 `0.99964`，
  (\|\Delta H\|_F/(4\|\Delta A\|_F)) 不超过 `0.25301`。
  候选 Agent 更新范数 `0.23929` 和 LoRA 更新范数 `0.05`
  均被零容差 probe 拒绝；两者的参数漂移与 OGD rank 变化均为 0，统一
  Agent 的版本增量也为 0。LoRA 桥本身不维护参数版本计数。
  这些是机制认证，不进入任务性能对比。
- `analysis/mqr_solver_certification.py` 扫描线性/tanh、三种 $\alpha$、三种步数和
  两种初始化尺度。所有正向与伴随 `residual/alpha` 上界成立；但
  $\alpha=0.03,K=20$ 的平均 transition-gradient 相对误差为 `0.3721`，
  $K=200$ 后仍为 `9.51e-4`。相反，$\alpha=0.10,K=200$ 和
  $\alpha=0.30,K=60$ 分别达到 `2.65e-10` 与 `2.77e-10`。这实证否定了
  “固定 20 步总是近似精确”，支持按 residual 认证并在失败时拒绝更新。
- `analysis/mqr_sinkhorn_fair_compare.py` 在 float64、$N=6$ 的匹配梯度问题上得到
  Cayley 拉回相对误差 `1.17e-15`、收敛 Sinkhorn 隐式拉回 `1.97e-15`。
  两者都通过梯度对拍；该结果只建立公平数学基线，不证明 Cayley 在任务性能或
  端到端速度上占优。
- Digits A→B 单遍流（3 seeds，batch 16）中，共享单环 A 遗忘为 `17.96±2.92` 个百分点；相同 6,880 参数的 OGD-64 为 `14.63±3.32` 个百分点，平均降低约 18.6%，但 B 最终准确率从 `89.38%` 降到 `88.21%`，且一个种子没有改善。显式双环把 A 参数漂移和遗忘都压到精确 0，但参数翻倍，B 最终准确率仅 `85.25%`，说明隔离牺牲正迁移。
- OGD 投影对白化历史子空间的最大残差内积约 `1.9e-8`，但 rank-64 稠密基占 `1,761,280` 字节，约为 6,880 个 fp32 参数本身的 64 倍。这证明数学机制工作，也暴露了直接扩到 2B 的内存不可行性。
- 现有 CIFAR-100 checkpoint：epoch 499，约 1.968M 张量参数，best test accuracy `59.98%`，train accuracy `96.06%`，test loss `1.6808`。
- 三次记录为 `56.00/59.18/59.98`，总体均值 `58.39%`；但缺少同编码器、同参数、同训练配方的 ViT-only/MLP-head/LoRA-head 对照。
- 该 checkpoint 使用历史错误的酉更新训练，不能验证本次修正后的算法；较高准确率很可能主要来自 ViT、注入和读出，需要重跑消融。
- 记录称配置 500 epoch，但曲线元数据仅有 100 个 logged epochs；论文图注与 checkpoint 需统一。
- 新的 5-seed A→B→A 反转流已在 288–320 个可塑参数、不超过
  4096 B 在线张量状态上限下比较 GRU、fast-weight、LoRA、OGD-LoRA 和
  equal-byte replay。双专家 MQR+OGD 的 A-after-B 下降为 `0.0 pp`，LoRA
  为 `36.75 pp`；遗忘减少的配对 bootstrap 95% CI 为
  `[34.00,39.50] pp`。但 MQR 全过程平均阶段准确率比 LoRA 低
  `8.58 pp`，延迟为 `36.31 ms` 对 `0.48 ms`，总 sidecar 参数为
  1354 对 300。因此预注册 `independent_competitive_advantage_established`
  仍为 false；该实验建立了正确路由条件下的稳定性机制，没有建立等总参数/
  FLOP 的总体优势。
- MiniCPM5-1B/Sayuri 机制实验在 RTX 3070 Laptop 上以约 2.20 GB 峰值 CUDA 显存运行。24 步单盘流完成一局，held-out loss 从 `3.289` 降到 `3.170`；48 步、8 个重复基础局面流的首/末四分位 loss 为 `3.149→2.249`，回放 masked 教师一致率为 `75%`，8 个规则抽查局面零不一致。
- 相同 seed 的 48 步 LoRA/no-LoRA 对照逐步 loss 平均绝对差仅 `2.17e-5`。这验证了 LoRA 梯度链和 checkpoint 路径，但没有证明 LoRA 的额外收益；短流改善主要来自 MQR，且单 seed、小 held-out 集不足以支持泛化结论。
- 3-seed 真实干扰 Digits 中，多尺度外部 oracle 写门为
  `73.44±5.16%`，逐位同初始化无门为 `20.10±1.83%`，配对改善
  `53.33±6.17 pp`；同稀疏度随机门为 `18.13±1.13%`。这证明受控写入
  可保护状态，不证明门已学得。多尺度相对单慢环仍为 `-2.81±4.33 pp`。
- 新的 24-run、3-seed 辅助事件门实验中，44 参数多尺度学习门为
  `70.83±6.55%`，无门为 `19.86±1.68%`，门固定不更新为
  `21.53±0.87%`；learned 相对 ungated 的配对效应为
  `+50.97±6.74 pp`，oracle 仅再高 `1.94±1.58 pp`。控制器 held-out
  cue/non-cue 二值准确率为 100%，但它只读取显式 marker 并在每帧后收到
  即时辅助标签，不是自主重要性发现。10%/25% false-open 分别使 oracle
  准确率下降 `20.28±5.37`/`33.75±7.23 pp`；约 10% false-close 下降
  `7.36±0.87 pp`。等状态单慢环仍高 `2.78±1.20 pp`。
- 新的 marker-free、随机目标位置 Digits 实验中，49 参数 OGD 未来收益门为
  `72.00±6.24%`，普通收益门为 `57.67±27.23%`，不更新门为
  `10.33±3.79%`，全写为 `8.67±3.79%`，同期望稀疏度随机写为
  `17.00±3.00%`；OGD 门相对不更新门/全写的配对提升为
  `+61.67±5.03`/`+63.33±9.50 pp`，并与因果 novelty 阈值及位置 oracle
  相同。目标/干扰的实际平均收益为 `+1.097±0.080`/`-0.463±0.120`，预测
  均值符号正确；但逐票据正收益 AUPRC 仅 `75.19±2.89%`。普通门相对 OGD
  的 held-out 差异完全由一个 seed 的误写塌缩造成，不能视为稳定泛化优势。
  该协议把“唯一目标”
  统计性地暴露为新颖事件，因此证明延迟任务收益可以学得可观测写入规律，不证明
  任意未来相关性发现。该 v5 单任务协议本身没有与基线比较；v6 反转协议的
  同上限对照不能追溯性地证明 v5 任务上的优势。

- MiniCPM5/Sayuri v2 在 5 个新 seed、4 个更新归因条件上每条件训练 8 局、
  配对评估 8 局。掩码后目差改善为 `+5.75`，Student-t 95% CI
  `[-2.25,+13.75]`；原生 probe loss 恶化 `0.209`，pass Brier 恶化
  `0.0087`，两者在五个 seed 上全部变差。学生回合未掩码合法率仍约
  10%。MQR+LoRA 与 MQR-only 离散行为相同；LoRA-only 参数有漂移但对局不变。
  这证明 MQR 动作头确实造成在线行为变化，但没有证明内部围棋规则学习、
  稳定棋力提升或 LoRA 增益。
- 修正四头联合概率后，10×10、5-seed 缩短 pilot 使用无损 306 维编码和任意尺寸
  ko/superko 配对。after-task-B 动态对局有效合法率为 MQR `0.0153`、identity
  `0.0277`、GRU `0.0552`、fast-weight `0.0215`；平均回报为
  `-17.125/-16.825/-16.425/-16.975`，历史焦点准确率全部为 `0.500`。
  参数/MAC 比通过 5% 门，但持久状态比为 `5.333`。所以该 pilot 的决策为
  `mqr_effective=false`，并表明固定稀疏 probe 上的 1.0 合法率不能代表动态对局。
  旧 10×10 文件直接拼接 pass 与 placement logits，指标被 pass 虚高，已从证据
  链排除。该短 pilot 不再承担正式结论。
- 后续 10×10/13×13 正式实验各用 10 seeds，加入 identity、GRU、
  LSTM、fast-weight、LoRA、OGD-LoRA、replay-LoRA 和隐式 Sinkhorn。
  所有方法均分配 48 B 循环状态接口，参数/前向/更新比在两尺度都
  小于 1.05。MQR task-A held-out loss 降幅为
  `2.8331 [2.7121,2.9531]` 和 `3.0168 [2.9378,3.0929]`，
  证明在线权重更新有效；但两尺度的历史焦点均为 `0.500`，
  13×13 replay-LoRA 在动态合法率和回报上显著超过 MQR。
- schema-v2 机制资格实验用 10 seeds、长度 12 和三个重复 query。
  cyclic MQR 历史准确率为 `0.5375 [0.5119,0.5631]`，identity
  为 `0.5875 [0.5493,0.6257]`；配对差为
  `-0.0500 [-0.07890,-0.02110]`。未来损失到注入、转移和最早输入的
  梯度均非零，但 identity replacement 不伤害性能，证明的是图连通而不是
  环的行为贡献。
- 48/192/768 B 等状态容量扫描中，三档摊销资源门均通过，
  MQR-minus-identity 准确率仍为 `-0.02875/-0.03250/-0.07125`；
  单次转移刷新峰值成本比为 `1.1596/1.7122/2.3110`。这排除了
  “只是容量太小”的解释。

因此当前证据证明了结构正确性、在线事务语义、局部 OGD 约束、显式环隔离以及
正确 marker-free 上下文路由下的专家稳定性。它仍不能证明 MQR 在总资源、平均
准确率和延迟上优于 LoRA，也不能外推为动物式一般学习。

## 10. 关于“一般动物学习能力”

该架构最多有希望模拟下列局部能力：稳定的递归状态、快速奖惩关联、上下文专用记忆模块、快慢权重和有限的增量学习。它尚缺少动物学习的关键组成：主动探索、自监督世界模型、稀疏奖励归因、情景回放与巩固、结构生长/剪枝、多模态闭环、身体与环境交互、动机和长期信用分配。

此外，当前伴随环仍需要全局损失梯度和权重相关的反向算子；“不使用 BPTT”不等于生物可实现。更稳妥的研究目标是：

> 先证明 2B 冻结主干上的环库能以低显存实现单遍快速适应，并在受控任务序列上比同参数 LoRA 少遗忘；不要直接宣称一般动物智能。

## 11. 建议实验路线

1. **正确性门（已完成）**：固定 seed 对拍精确公式、直接伴随与 Autograd；继续把 residual、认证失败率和梯度误差作为每个新配置的前置门。
2. **基础学习门（未通过）**：正式 Go 证明所有核心都会降低训练/
   held-out loss，但历史焦点未超过 0.5。后续先要求 marker-free 写门形成
   显著 event--distractor 间隔，再扩大任务。
3. **等资源消融（已执行，结果为负）**：正式 Go 补齐九种方法；
   48/192/768 B 容量扫描又单独对拍 cyclic 与 identity。当前不支持环拓扑优势。
4. **围棋复验（有条件暂停）**：10×10/13×13、10 seeds 已完成。只有在
   合成机制任务上击败 identity、identity replacement 显著有害且自主门学会
   选择写入后，才重启更长 Go 训练。
5. **持续学习**：Split CIFAR-100 或 class-incremental 流，单遍训练；报告平均准确率、最大遗忘、forward/backward transfer、在线 regret。
6. **小语言模型试验**：正式核心门通过后，再在冻结模型最终隐藏层外挂，比较 LoRA、OGD-LoRA、replay、单环和环库；慢速 LoRA 与短策略 RL 分阶段启用以保持归因。
7. **扩到 2B**：推荐 `N=64/128,r=8,K=1~3,Top-1`，先只更新输入/输出投影，慢速解冻 `U`。
8. **动物式子能力**：分别评估一次性联想、奖励反转、上下文切换、延迟匹配、睡眠 replay，而不是使用一个模糊总称。

建议的首要成功标准不是单任务准确率，而是：在相同外挂参数、相同在线样本数下，旧任务遗忘显著低于 LoRA，同时每 token 延迟和显存增量可接受。

## 12. 可复现验证

```bash
python3 test_mobius_model.py
python3 test_online_learning.py
python3 test_temporal_mqr.py
python3 test_utility_mqr.py
python3 test_unified_mqr_agent.py
python3 test_go_minicpm.py
python3 analysis/mqr_proof_verify.py
python3 analysis/mqr_sidecar_safety_verify.py
python3 analysis/mqr_solver_certification.py
python3 analysis/mqr_sinkhorn_fair_compare.py
python3 analysis/temporal_mqr_interference_verify.py
python3 analysis/temporal_mqr_learned_gate_verify.py
python3 analysis/temporal_mqr_utility_verify.py
python3 analysis/temporal_mqr_reversal_verify.py
python3 analysis/minicpm_go_real_games_v2_verify.py
python3 experiments/online_digits_principle.py
python3 experiments/temporal_mqr_interference_digits.py
python3 experiments/temporal_mqr_learned_gate_digits.py
python3 experiments/temporal_mqr_utility_digits.py
python3 -m py_compile mqr/*.py experiments/*.py train_mobius_cifar100.py
```

最终判断：**将多个小环连接到冻结模型的工程链路已经可运行，结构收缩、合法
Cayley 更新、有限轨迹信用和 OGD 一阶隔离也可验证；但 10×10/13×13
正式围棋、10-seed 机制资格和三档容量扫描一致否定了当前 MQR 的独立
算法优势。它不是 LoRA 的天然替代，更不能仅凭当前结构推出 2B 收益或
一般动物学习能力。**
