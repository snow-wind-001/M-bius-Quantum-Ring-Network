# MQR 环状 Sidecar：数学修订、实现方案与验证门槛

> 状态：2026-08-29 已按 P0--P2 实施，并完成正式 10×10/13×13、
> schema-v2 机制资格与三档等状态容量扫描。本文保留“先固定口径、
> 再修代码、最后验证”的实施记录；任何 smoke/pilot 均不替代正式证据。

## 1. 目标域与当前结论

目标不是用 MQR 替代冻结小模型，而是构造一个在线侧车：冻结或慢速更新主干
\(f_\phi\)，MQR 在严格 `predict-before-update` 事务中读取隐藏表示，维护快慢状态，
选择性写入，并输出 residual feature、prefix 或低秩适配信号。Cayley 环参数只在慢
时间尺度刷新；读出、写门和小型投影可快速更新。

当前可支持的结论是：MQR 提供解析的酉/双随机约束、统一收缩率和可审计的在线
事务，但尚未证明其核心优于 identity、GRU、fast-weight、LoRA 或 replay。线性
平衡态 MQR 严格坍缩为低秩线性映射；独立价值只能来自非线性、时序状态、选择性
写入、路由及遗忘—可塑性权衡，而且必须由等资源实验归因。

## 2. 对现有数学质疑的裁决

| 质疑 | 裁决 | 必要调整 |
|---|---|---|
| 式 (13) 不能覆盖 \(\mathfrak u(n)\) | 不成立。\(\mathfrak u(n)=\mathfrak{so}(n)\oplus i\operatorname{Sym}(n)\)，实维数为 \(n^2\) | 论文补维数证明；代码增加最小 \(n^2\) 坐标，保留旧 checkpoint 模式 |
| Cayley 覆盖整个 \(U(n)\) | 对单个 chart 不成立；但全局相位可避开 \(-1\) 谱，且不改变模平方 | 不声称 Cayley 满射 \(U(n)\)；改为严格证明 Cayley--\(|\cdot|^2\) 复合覆盖全部 unistochastic 集，并监控逆坐标条件数 |
| 参数化梯度丢失一族方向 | Cayley 在有限 \(A\) 上局部满秩；主要退化来自 \(U\mapsto |U|^2\) | 报告模平方映射 Jacobian 秩；特别标注 \(U=I\) 时 \(dH=0\) |
| 非线性伴随仍是 \(p=\alpha g(I-qH)^{-1}\) | 不成立 | 使用 \(p=\alpha g(I-qDH)^{-1}\)，或列式 \((I-qH^\top D)p^\top=\alpha g^\top\) |
| \(qh^*dH^\top\) 被当作二阶项截断 | 不成立；它是一阶项，隐式伴随会消去 \(dh^*\) 而保留该项 | 澄清推导；用直接线性求解与完整 autograd 对拍 |
| 固定 \(K\) 的隐式梯度“近似精确” | 无统一保证 | 正向和伴随均返回 residual、迭代数、误差上界与认证状态；认证失败默认不提交更新 |
| 非负性是收缩/隐式梯度前提 | 不成立 | 非负性只用于齐次双随机传播的 \(\ell_1\) 质量解释；有符号 sidecar 不复用该命题 |
| 应强制“最小更新量”防止 OGD 关门 | 不安全 | 不强迫破坏保护子空间；报告 retained norm，并通过新环、容量回收或受控巩固恢复可塑性 |

## 3. 时变环与 Sidecar 的严格适用域

令每个在线时刻

\[
T_t(h)=\sigma(qH_t h+\alpha J_t),\qquad q=1-\alpha,
\]

其中 \(H_t\) 任意双随机，\(\sigma\) 在 \(\ell_\infty\) 下 1-Lipschitz。对经历同一
\(H_t,J_t\) 序列的两个状态，有

\[
\|h_t-\tilde h_t\|_\infty\le q^t\|h_0-\tilde h_0\|_\infty.
\]

因此，收缩率对时变 \(H_t\) 一致成立。需要修正的不是逐步稳定性，而是“共同固定
点”的说法：每个冻结时刻的 \(T_t\) 有唯一固定点，但变化的 \(T_t\) 通常没有同一个
平衡态。若 \(h_t^*\) 是瞬时固定点，则跟踪误差满足

\[
\|h_t-h_t^*\|_\infty
\le q\|h_{t-1}-h_{t-1}^*\|_\infty
+q\|h_t^*-h_{t-1}^*\|_\infty.
\]

所以在线更新还必须控制瞬时固定点漂移；结构收缩不能替代参数 trust region。

对于收敛的非线性固定点，\(D=\operatorname{diag}(\sigma')\)，精确伴随为

\[
(I-qH_t^\top D)p^\top=\alpha g^\top,
\quad
\nabla_HL=\frac q\alpha(Dp)(h^*)^\top.
\]

若迭代残差为 \(r=\|T(h)-h\|_\infty\)，由收缩性可认证
\(\|h-h^*\|_\infty\le r/\alpha\)。小 \(\alpha\) 不必使缩放伴随 \(p\) 爆炸，
但会放大求解误差、增加所需迭代，并通过 \(q/\alpha\) 放大 transition gradient；
因此必须同时报告 residual 和最终梯度误差，不能只报告酉性误差。

## 4. Sidecar 的最低安全契约

1. **原子因果事务**：输出必须来自 \(\theta_t\)，反馈只能生成
   \(\theta_{t+1}\)；ticket 绑定 stream、主干版本、KV/feature 版本和单次消费状态。
2. **默认无扰动**：冻结主干不被侧车初始化改变。采用零初始化 residual readout，
   而不是要求环状态非负；后者会限制有符号特征修正。
3. **求解认证**：启用 tolerance 时，正向或伴随未收敛则不提交参数、OGD 记忆或
   外部 encoder 梯度；研究者可显式允许 `allow_inexact_update`，但结果必须标记。
4. **双时间尺度**：每 token 可更新状态/门/读出；稠密 Cayley 只在块边界慢速更新
   并缓存 \(U,H\)。环宽必须是小子空间 \(N\)，不能是模型参数维。
5. **有界变化**：保留完整更新 trust region；监控 state/injection 范数、solver
   residual、transition Jacobian 灵敏度和旧域 probe。性能守恒需要独立 no-write
   或 rollback gate，不能从酉性推出。
6. **正性声明隔离**：只有实验显式使用非负状态不变集时，才可引用
   \(\ell_1\) 质量守恒；语言模型 residual sidecar 默认是有符号空间。

## 5. 本轮代码修改

### P0：固定点与隐式梯度认证

- `mqr/ring.py` 增加可选 forward tolerance、最小/最大步数及缩放 residual。
- `RingState` 携带 forward residual、\(r/\alpha\) 误差界、迭代数和
  `converged`；旧的 `RingState(h=...)` 与固定步数行为保持兼容。
- 伴随迭代提供同构诊断，并增加小环非线性直接 solve oracle。
- `eqprop_update_step` 返回完整 solver 字段。只要请求了认证，任一路径失败且未
  显式允许非精确更新，就原子跳过全部参数、OGD 和外部梯度更新。

### P1：Cayley 坐标与 Sidecar 初始化

- `mqr/unitary.py` 保留 `projected` 模式的 `A_real/A_imag` 旧布局；新增
  `minimal` 模式，以标准正交基存储 \(n(n-1)/2+n(n+1)/2=n^2\) 个实坐标。
- 提供坐标梯度拉回、两模式之间的 \(A\) 导入、有效自由度/冗余度和
  \(U\mapsto |U|^2\) 局部 Jacobian 秩诊断。
- 环构造器增加可选坐标模式与零初始化 readout，供 residual sidecar 使用；默认
  值完全保持现有 checkpoint 和前向行为。

### P2：证据脚本

- 新增无网络 float64 认证脚本，扫描 activation、\(\alpha\)、步数和初始化尺度；
  对比 unrolled autograd、迭代 implicit 和直接伴随 oracle。
- 记录 forward/adjoint residual、认证误差界、各参数块相对梯度误差、时间和失败率。
- 正式检验时变 \(H_t\) 的统一收缩，而不是错误地套用共同固定点定理。

## 6. 实验与停止条件

| 优先级 | 实验 | 核心指标 | 失败/停止条件 |
|---|---|---|---|
| P0 | float64 非线性梯度对拍 | residual、相对误差、余弦、认证率 | 已认证样本仍大幅超出残差推导界，停止行为实验并修数学链 |
| P0 | identity 点与泛型点 Jacobian | rank、奇异值、off-diagonal mass | 把 identity 初始化用于学习 \(H\) 却无破对称机制 |
| P1 | sidecar no-op/事务测试 | 初始化 drift、跳过更新零漂移、ticket 版本 | 任何未认证更新或预测后写泄漏 |
| P1 | 等资源机制任务 | accuracy/regret、遗忘、参数/状态字节、FLOP、P50/P95 | MQR 未同时超过 identity 与至少一个非 MQR 核心 |
| P2 | 围棋规则/历史任务 | 未掩码合法率、pass Brier、value、回报、旧域退化 | 仅 masked 指标改善或原生 probe 持续恶化 |
| P2 | 大棋盘 | 至少 10×10；推荐 13×13 主结果，10–20 seeds | 3×3/5×5 只能标注机制 smoke，不进入有效性主表 |

围棋主比较必须使用相同路由、损失、反馈次数、OGD、可塑参数、总 sidecar 参数、
状态字节、测得 FLOP/延迟，并包含 identity、GRU、LSTM、fast-weight、LoRA、
OGD-LoRA、equal-byte replay 和 Sinkhorn。这一门已在 10×10/13×13、每尺度
10 seeds 的正式实验中实施；分配资源门通过，但性能门失败。

## 7. 论文结论门

只有同时满足以下条件，才能把 MQR 从“稳定、可审计的机制原型”提升为“独立有
竞争力的在线 sidecar 算法”：

1. MQR 在同资源下同时击败 identity 和至少一个 GRU/fast-weight 核心；
2. 未掩码合法率、任务回报/next-token loss 与跨任务遗忘共同改善，而非只改善
   路由后的 masked 指标；
3. 增益不能由专家隔离、OGD、额外状态、额外参数或更多反馈解释；
4. 至少 10–20 个正式 seed 的区间不跨过预注册的最小实际效应；
5. 报告 solver 未认证率、近 identity 退化、延迟、显存和失败案例。

在此之前，合理表述仍是：MQR 的数学约束对小维在线侧车有用，但“提升小模型
智能”与“模拟一般动物学习”是尚待分解验证的研究目标，不是当前结论。

## 8. 实施记录

本轮修改严格按上述顺序完成，没有以实验结果反向改写验收门槛。

- `mqr/ring.py` 已加入正向/伴随 residual、相对 residual、迭代数、
  `residual/alpha` 误差证书和非线性直接伴随 oracle。请求认证时，任一求解器
  未收敛会原子拒绝参数、OGD 记忆和外部特征梯度更新；旧固定步数 API 保持兼容。
- `mqr/unitary.py` 已加入严格 $N^2$ 自由度的 `minimal` Cayley 坐标，同时保留
  旧 `projected` checkpoint 布局。两种坐标可导入同一 $A$，并可诊断
  $A\mapsto U\mapsto |U|^2$ 的局部 Jacobian 秩和近单位退化。
- `mqr/baselines.py` 已加入 log-space Sinkhorn 基线及收敛矩阵缩放的隐式梯度，
  用于避免把 straight-through 或截断反传当作公平对照。
- residual sidecar 支持零初始化读出，初始输出严格无扰动。10×10 Go 路径改为
  306 维无损特征，ko/superko 配对支持任意 $N\geq3$。四头策略改用
  $P(i)=P(\neg\mathrm{pass})P(i\mid\neg\mathrm{pass})$，不再拼接不同语义的
  pass 与 placement logits。

## 9. 首轮验证与决策

七个测试入口共 111 项通过：`test_mobius_model.py` 24 项、
`test_online_learning.py` 7 项、`test_temporal_mqr.py` 31 项、
`test_utility_mqr.py` 11 项、`test_unified_mqr_agent.py` 12 项、
`test_competitive_go.py` 4 项和 `test_go_minicpm.py` 22 项；
`analysis/mqr_proof_verify.py` 亦通过。

Float64 求解扫描表明，固定步数不能替代 residual 认证。以四个线性/非线性与
初始化尺度组合的均值计，\(\alpha=0.03,K=20\) 的 transition-gradient 相对误差
为 `0.3721`，增加到 `K=200` 后仍为 `9.51e-4`；
\(\alpha=0.10,K=200\) 为 `2.65e-10`，\(\alpha=0.30,K=60\) 为
`2.77e-10`。扫描中的所有正向和伴随 `residual/alpha` 上界均成立。

匹配的 $N=6$ 数值对拍中，Cayley 拉回相对误差为 `1.17e-15`，收敛
Sinkhorn 隐式拉回为 `1.97e-15`。这验证了两条梯度链，不构成 Cayley 的任务性能
或速度优势；本机 CPU 小矩阵计时也不足以外推到加速器。

修正 pass 联合概率后的 10×10、5-seed 缩短 pilot 得到以下 after-task-B 对局
有效合法率：MQR `0.0153`、identity `0.0277`、GRU `0.0552`、fast-weight
`0.0215`；平均回报分别为 `-17.125/-16.825/-16.425/-16.975`，历史焦点准确率
全部为 `0.500`。参数和估算 MAC 比为 `1.0359/1.0307`，但状态比为 `5.333`。
因此 `mqr_effective=false`。旧
`unified_temporal_mqr_go_10x10_5seed_pilot.json` 使用错误 pass 组合，不能作为
证据；权威 pilot 是
`unified_temporal_mqr_go_10x10_factorized_5seed_pilot.json`。

下一阶段不是直接扩大宣传口径，而是先让所有核心在动态 held-out 局面达到非零
placement 学习和显著高于随机的有效合法率，再解决等状态资源门，并运行至少
10--20 seeds 的 10×10/13×13 正式比较。该比较仍须补齐同协议 LoRA、OGD-LoRA
与 equal-byte replay；在这些门通过前，不启动长期策略 RL 或慢速 LoRA 归因实验。

## 10. 最终一致性审计与补充修改

在论文数字、结果 JSON 和验证器的逐项对拍中，发现有效性门的两个
legality 字段实际取自稀疏 task-A probe，而论文表格和结论解释的是无规则
mask 动态对局。两者分布差异极大：MQR 的 probe useful legality 均值为
`0.9500`，动态对局却只有 `0.0153`。这是证据语义不一致，必须在继续
实验前修正，不能仅靠文字解释。

补充修改按以下顺序执行：

1. 将 `unmasked_legal_rate` 和 `useful_legal_placement_rate` 的有效性
   比较统一改为 `game_returns.<final_stage>` 动态对局指标；回报继续取自
   同一评估轨迹，历史焦点仍取自输入相同的 task-B probe。
2. 在 JSON 中显式写入每个门指标的来源路径，并让 fail-closed 验证器
   从同一路径重算均值差，防止文档与代码再次漂移。
3. 资源门记录具体 ratio limit；宽松 9× 状态上限只能标注为历史诊断，
   不得参与 `mqr_effective` 决策。唯一决策资源门是 1.05 的严格等参数/
   等状态/等 MAC 门。
4. 修正“所有核心 placement 教师一致率严格为零”的模糊表述。task-B
   历史 probe 上确实全为零；动态对局上 MQR/identity 为零，GRU 与
   fast-weight 分别约为 `0.0046/0.0016`，仍属基础 placement 学习失败，
   但不应舍入成“全零”。
5. 重算权威 10×10 JSON，再运行验证器、全部回归、证明脚本、
   `git diff --check` 和论文 PDF 编译。修正后仍以预先登记门决定结论，
   不允许因数值方向而更改门槛。

上述补充修改已实施并从原始轨迹重算。动态 useful-legality 的 MQR 配对均值差
相对 identity、GRU 和 fast-weight 分别为 `-0.01245/-0.03994/-0.00620`；
95% pilot bootstrap 区间分别为 `[-0.02814,-0.00005]`、
`[-0.07089,-0.01543]` 和 `[-0.01240,0.00000]`。严格状态资源门仍失败，
所有历史焦点仍为 `0.500`，因此修正后的决策仍为
`mqr_effective=false`。该修正消除了证据语义错位，没有改变或放宽结论门。

## 11. Phase II：安全 Sidecar 闭环与候选提交

### 11.1 数学契约

语言模型 sidecar 默认在有符号隐空间中工作。对第 (j) 个环，令
(c_j=1-\lambda_j)，外部强迫为 (b_{j,t})。因为 identity 和双随机
转移在 (\ell_\infty) 下的诱导范数为 1，且当前状态激活均为
1-Lipschitz 且在零点取零，逐步有

\[
 \lVert h_{j,t}\rVert_\infty
 \le c_j\lVert h_{j,t-1}\rVert_\infty+\lVert b_{j,t}\rVert_\infty.
\]

实现必须同时返回该解析上界、实测状态范数和边界违反量。
它不依赖非负性；只有显式锥不变实验才另外报告 (\ell_1)
质量解释。

对两个斜 Hermitian 坐标 (A,B)，Cayley 映射满足

\[
 \lVert U(A)-U(B)\rVert_F\le 2\lVert A-B\rVert_F,
 \qquad
 \lVert |U(A)|^2-|U(B)|^2\rVert_F\le4\lVert A-B\rVert_F.
\]

因此每次慢速环更新要报告实际 (A/U/H) 漂移和上界；该界只
控制结构漂移，不替代主干输出守恒测试。

### 11.2 代码修改顺序

1. `mqr/temporal.py`：增加 signed-state 逐步证书，记录 forcing、前后
   状态 (\ell_\infty) 范数与 violation，不改变默认前向返回值。
2. `mqr/unitary.py`：增加 Cayley 有限漂移诊断，数值检查两个理论上界。
3. `mqr/agent.py`：将参数更新改为“快照→候选更新→本地输出/
   状态/策略 KL/转移漂移审查→提交或原子回滚”。Cayley 参数仅在
   `transition_update_interval` 到期时进入活跃集。
4. `mqr/minicpm.py`：LoRA 外部梯度接口增加候选更新范数和可选
   constancy closure。当 probe KL/输出漂移越界时，同时恢复 LoRA
   和 OGD 记忆。
5. OGD 的 `retained_norm` 只产生可塑性告警；不强制注入已被
   保护子空间拒绝的更新。可选处置是分配新环、在明确策略下回收基，
   或跳过慢速巩固。

### 11.3 验收与证据边界

- 单元测试要构造 signed state，检查解析上界在 identity 和
  unistochastic 两种核下成立，并注入故意损坏的候选更新验证
  参数、版本、OGD 基和已提交快状态均零漂移。
- 慢时标测试必须显示读出/注入可每次更新，而 Cayley 只在指定
  间隔更新；控制组使用同一调度器。
- constancy 是安全门，不是性能证据。本轮确定性/smoke 结果只能
  证明交易语义和数值界正确，不得改写已有 `mqr_effective=false`。
- 独立竞争力仍需 10--20 seeds、至少 10×10 长训练和 13×13 扩展复验，以及同参数、
  同状态字节、同 FLOP 和同旧能力退化预算的 identity、GRU、
  fast-weight、LoRA、OGD-LoRA、equal-byte replay 和 Sinkhorn 对照。

### 11.4 实施与验收结果

Phase II 已按上述顺序完成。`mqr/temporal.py` 新增 signed-state
逐步证书和零扰动 `TemporalMQRSidecar`；`mqr/unitary.py` 新增
(A/U/H) 有限漂移诊断；`mqr/safety.py` 定义策略 KL、输出漂移和
residual 指标。统一 Agent 现在使用候选提交，并以
`transition_update_interval` 明确分离快参数与慢 Cayley 坐标。
MiniCPM LoRA 外部梯度接口可接收原生 logits constancy closure。
统一 Agent 越界时会恢复参数、OGD 基和版本，并撤销外部梯度授权；LoRA 桥
恢复适配器参数与 LoRA OGD 基，但自身不维护版本计数。LoRA 桥只能观测
policy KL/输出漂移；若传入只能由统一 Agent 观测的 recurrent-state 或
Cayley-transition 预算，接口会 fail closed，而不是静默忽略。

慢时标与 OGD 的组合没有使用“投影后直接清零”这个会破坏正交性
的近似。`OrthogonalGradientMemory` 在当前允许坐标子空间内解
(z=Mw-MQ^\top(QMQ^\top)^+QMw)，使禁用的 Cayley 坐标严格为零，
同时保留 (Qz=0)。低 `retained_norm` 只返回可塑性告警，没有加入
强制最小更新量。

七个规范测试入口现为 `24+7+31+11+12+4+22=111` 项，全部通过。
`analysis/results/mqr_sidecar_safety_certification.json` 记录了 20 个
signed-state 扫描和 40 个 Cayley 漂移扫描，全部认证；状态界最大
violation 为 0，最大 unitary/transition 界比分别为
`0.999637/0.253003`。Agent 与 LoRA 的故意越界候选在回滚后参数
漂移和 OGD rank 均为 0，Agent 版本增量也为 0。所有旧结果验证器和 10×10 权威 pilot
验证器再次通过。

这些结果将 sidecar 从“结构上似乎安全”推进到“单步数值界与事务
原子性可认证”，但它们不改变任务结论：
`mqr_effective=false`，独立算法优势仍未成立。

## 12. 应用定位资格验证与正式证据审计

为核实“冻结小模型的受控在线个性化、会话记忆和错误修正层”这一定位，新增
`experiments/frozen_sidecar_application_qualification.py`。它不是算法排名实验，
而是 10-seed 的最小应用契约测试：冻结主干和原生头；MQR 初始 residual 为零；
两个会话对同一 query 需要相反输出；每个 seed 接收 100 次 predict-before-update
反馈；只更新读出；一个旧域 probe 通过精确读出零空间投影保护。

10/10 seeds 均得到：冻结主干个性化准确率 `0.500`，保留环状态后 `1.000`，
query 前清空状态后回到 `0.500`；错误会话 B 的平均 NLL 从 `0.9038` 降到
`0.2780`。主干、固定 sidecar 参数和转移漂移均为 0，受保护 probe 最大 policy
KL 为 `2.86e-16`，residual L2 不超过 `2.0`，状态界全部认证。10 个故意越界
候选均被拒绝，回滚后参数和输出误差为 0。counterfactual 标签测试的更新前输出
差为 0；交错/独立 stream 的状态和 logits 误差也为 0。

因此该定位在**合成机制层**成立；它仍不是实际 MiniCPM 个性化收益。真实
MiniCPM smoke 只证明冻结参数 probe 零漂移和链路连通。通用
`TemporalMQRSidecar` 的候选事务目前由调用方实现，只有统一 Go Agent 已内建完整
候选提交，后续实用化应统一这一包装层。

`analysis/mqr_application_positioning_verify.py` 对三类证据做 fail-closed 审计。
该审计当时使用的 10×10 证据只有 5 seeds，同协议缺 LSTM、LoRA、
OGD-LoRA、equal-byte replay 与 Sinkhorn，状态比 `5.333` 失败。这些是
历史定位审计的限制；后续正式 10×10/13×13 实验已补齐完整基线与
等状态资源，但 `mqr_effective` 仍为 false。

最终分层结论为：

1. 冻结模型 sidecar 的应用机制资格通过；
2. 真实小模型的应用增益未建立；
3. 通用能力增强和动物式学习未建立；
4. MQR 的独立算法优势未建立；
5. 10×10/13×13、10 seeds 与完整基线的正式竞争门已执行；
   资源门通过，历史利用、遗忘和联合性能门失败。

完整协议、数值和应用边界见
`analysis/mqr_application_positioning_report.md`；机器可读结果为
`analysis/results/frozen_sidecar_application_qualification.json` 和
`analysis/results/mqr_application_positioning_audit.json`。

## 13. 正式机制、容量与大棋盘闭环

10×10 和 13×13 正式 Go 实验各使用 10 seeds，对照包含 identity、GRU、
LSTM、fast-weight、LoRA、OGD-LoRA、replay-LoRA 和隐式 Sinkhorn。状态
接口比为 `1.0000`，两尺度的参数/前向/更新最大比为
`1.0431/1.0304/1.0130` 和 `1.0398/1.0253/1.0127`。MQR 的
held-out loss 显著下降，但所有方法的历史焦点都为 `0.500`，
13×13 replay-LoRA 的动态合法率和回报显著更好。

为判断是否只是 Go 目标或 48 B 状态过小，又完成了 schema-v2、
长度 12、三次重复 query 的 10-seed 机制实验。cyclic MQR 历史准确率
`0.5375 [0.5119,0.5631]`，identity 为 `0.5875 [0.5493,0.6257]`；
配对 MQR-minus-identity 为 `-0.0500 [-0.07890,-0.02110]`。未来损失
对注入和转移的梯度都显著非零，但 identity replacement 不降低准确率，
反转准确率仅 `0.3925 [0.3219,0.4631]`。这将结论分成：
“信用链连通”为真，“环转移改善行为”为假。

48/192/768 B 等状态容量扫描中，摊销参数、状态、前向和更新门
三档均通过；但 MQR-minus-identity 准确率为
`-0.02875/-0.03250/-0.07125`，768 B 的 95% 区间全负。单次 Cayley
刷新峰值成本比为 `1.1596/1.7122/2.3110`，全部失败。因而
当前根因不是 Cayley 坐标覆盖缺口或单纯容量不足，而是模平方局部
退化、双随机混合对任务的归纳偏置不如 identity、自主写门近似常开以及
刷新峰值成本。详细数值、证据边界和后续停止门见
`analysis/mqr_mechanism_and_capacity_report.md`。

## 14. Phase III：从内容混合改为地址路由

Phase III 新增 `mqr/routing.py`，不改变旧 `MultiTimescaleMQR` checkpoint 或
历史结果。物理内容槽与逻辑地址分离；无写时内容严格零漂移，残差算子

\[
T_\varepsilon=(1-\varepsilon)I+\varepsilon T
\]

只路由地址。实现支持 identity、局部循环置换、稀疏 cyclic Givens、分块 Cayley
和稠密 Cayley，并分别报告前向、刷新、摊销和峰值解析成本。因为软残差循环在
(0<\varepsilon<1) 时最终扩散到均匀地址，运行时用 straight-through top-1
提交离散地址；当 (T) 是无固定点置换且 (\varepsilon>1/2) 时，前向地址每步
精确执行该置换。这个非线性投影不复用线性双随机状态定理，其反向梯度是有偏
surrogate，文档和结果均显式标注。

`BudgetedMultiTimescaleUtilityGate` 使用 16/64/128 三个未来收益、正负分区
平衡 replay、对偶阈值和逐前缀硬预算。校准输出包括 AUPRC、Brier、ECE、总写率、
事件/干扰写率间隔和预算违反。8 项新测试覆盖双随机性、残差公式、内容不变量、
512 步离散路由、写槽隔离、稀疏成本、硬预算和校准。

正式 10-seed 资格实验在 64 B 等状态下通过用户规定的机制门：MQR 在长延迟、
上下文反转和多次查询均为 `1.000`，显著超过 identity、GRU 和 fast-weight；
identity replacement 降幅为 `0.43379 [0.42028,0.44729]`，event/distractor
写率为 `1.000/0.000`，五轴 MQR/control 比均不超过 `1.05`。

但该任务使用已知循环置换，事件幅值也容易与干扰分离。直接固定环形缓冲区同样
得到 `1.000`，与 MQR 配对差为 0。因此新增两个不同结论位：
`mqr_route_mechanism_qualified=true`，但
`mqr_independent_algorithm_advantage=false`。下一阶段必须学习未知/变化拓扑、
匹配事件与干扰边缘分布，并击败直接可学习稀疏路由；在此之前不重新启动 Go/LLM
放大或策略 RL。完整证明与结果见 `analysis/mqr_routed_memory_report.md`。
