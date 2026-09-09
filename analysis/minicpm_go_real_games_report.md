# MiniCPM–MQR 多局围棋在线学习：因果验证与理论审计（含 v2）

> 更新：2026-08-25。最新主结果是
> `analysis/results/minicpm_go_real_games_v2_5seed.json`；旧的 3-seed 课程与
> prompt 对照仍保留为历史证据。本报告评估的是“冻结预训练表示 + 在线
> MQR 动作头”，不是 MiniCPM 语言头直接生成棋步。

## 0. 五种子 v2 更新

v2 使用 seed 2031–2035，比较 MQR+LoRA、MQR-only、LoRA-only 和
no-learning；每个条件训练 8 局、前后各评估 8 局，并使用 24 个落点目标
与 8 个 pass 目标的固定分层探针。探针每 2 局重测，实战始终使用原生
Sayuri 评估教师。

| 指标 | 训练前 | 8 局后 | 变化 | 5-seed Student-t 95% CI |
|---|---:|---:|---:|---:|
| 原生固定探针 loss | 3.273 | 3.482 | 恶化 0.209 | `[0.107,0.311]` 恶化 |
| 条件落点 loss | 3.227 | 3.270 | 恶化 0.044 | `[-0.087,0.175]` |
| 落点一致率 | 6.67% | 1.67% | -5.00 pp | `[-12.67,+2.67] pp` |
| pass Brier | 0.2328 | 0.2415 | 恶化 0.0087 | `[0.0056,0.0118]` 恶化 |
| 探针未掩码合法率 | 71.25% | 59.38% | -11.88 pp | `[-62.64,+38.89] pp` |
| 掩码后学生目差 | -10.08 | -4.33 | +5.75 | `[-2.25,+13.75]` |

探针总 loss 与 pass Brier 在五个 seed 上全部恶化；学生回合未掩码合法率
仅从 `8.96%` 到 `10.23%`。MQR+LoRA 与 MQR-only 的离散动作和配对目差
逐 seed 相同。LoRA-only 参数 L2 有约 `4.0e-4` 漂移，但配对目差变化全为
0，probe loss 变化绝对值小于 `1.5e-5`。所以当前可见行为更新来自 MQR
动作头；正的掩码后目差趋势既未达到统计显著，也不能解释为内部规则学习。
完整归因、反转对照与结果哈希见
[`mqr_reversal_go_v2_report.md`](mqr_reversal_go_v2_report.md)。下文 3-seed 结果应在该更新
下解读，不再作为最新 headline evidence。

## 1. 结论

当前系统已经证明三件事：权重确实在变，变化确实由预测后的反馈引起，而且能改变未参与更新的局面上的策略。它还没有证明“学会围棋规则”或“稳定提升棋力”。

- **原生 Sayuri 反馈**：3 个 seed 上，固定 probe loss 改善 `0.335±0.067`，原始合法率提升 `27.1±9.5` 个百分点，但配对实战目差反而变化 `-11.5±8.2`。后半段策略塌缩到 `pass`：监督更像教师，任务得分更差。
- **pass 课程优化**：在占用率 60% 前抑制 Sayuri 过早 pass，并暂停 pass 标签的权重更新后，三个 seed 的配对目差都改善：`+4.5, +1.0, +15.0`，均值 `+6.83±7.29`。对原生 Sayuri 交叉评估仍为 `+6.50±7.70`。但样本太小，95% t 区间仍跨 0；同时原始合法率和固定 probe 上的教师一致率没有改善。
- **规则 prompt**：正确规则相对 board-only 的交叉评估目差差值是 `+5.0±0.5`，但正确规则相对等结构的错误规则只是 `+1.17±3.21`，且有一个 seed 更差。因此当前只能说“附加自然语言前缀改变了冻结特征”，不能说“模型理解了正确规则”。
- **LoRA**：LoRA 的 L2 漂移约 `9.8e-5`，但 rules-online 和 MQR-only 在棋步、目差和 probe 指标上一致。当前可见学习几乎全部来自 MQR，没有 LoRA 增益证据。
- **冻结对照**：所有 seed 都满足 MQR/LoRA 零漂移、probe 完全一致、配对对局逐手一致，排除了评估噪声伪造的前后变化。

所以，最准确的当前结论是：**这是一个可运行的快速在线适配器，能改变真实闭环行为；但它还不是可靠的规则学习器、围棋策略学习器或动物级学习系统。**

## 2. 实际系统和参数边界

对每个局面 \(s_t\)，系统计算

\[
z_t=\operatorname{LN}(f_{\theta_0,\phi_t}(p(s_t))),\qquad
\ell_t=\operatorname{CE}(R_{\psi_t}(z_t),a_t^E),
\]

其中 `MiniCPM5-1B-AWQ-INT4` 的 880,092,672 个主干参数 \(\theta_0\) 在加载时一次解包为冻结 FP16 张量。可更新的 \(\phi\) 是末层 `q_proj/v_proj` 上的 43,008 个 FP32 LoRA 参数，\(\psi\) 是黑/白两个 MQR 的 45,312 个参数。这不是原生 INT4 训练，也不是语言头微调。

部署时学生动作使用外部合法着集合 \(A(s)\)：

\[
\tilde a_t=\arg\max_{a\in A(s_t)}q_{\phi_t,\psi_t}(a\mid s_t).
\]

因此实际落子的合法率恒为 100%，它是环境约束的结果。只有未加 mask 的 `raw_legal_rate` 能表明头部是否学会避免非法着。

Sayuri 以独立 GTP 进程运行，权重 SHA256 为 `31e31e6b8c3af59bc996470c91c262ea9aeb4fb1df6c8d2d5fafc320240bcb81`。峰值 CUDA 分配约 2.05 GiB。

## 3. 因果在线协议

学生执黑/白逐局交替，每局间保留权重、清空环的瞬时状态。每一手严格执行：

1. `preview_step` 用 \(\Theta_t=(\phi_t,\psi_t)\) 计算 logits，不修改参数、环状态、路由统计或 OGD 记忆；
2. 若轮到学生，先将旧策略的 masked 动作真正落到棋盘；
3. 保存的落子前局面再获得 Sayuri 标签；
4. `online_step` 用同一输入和旧状态提交 \(\Theta_t\to\Theta_{t+1}\)。测试要求 preview 和提交返回的更新前 logits 逐位一致。

Sayuri 回合也先让学生预测，再将教师动作作为在线示范。这是学生轨迹上的在线模仿学习，与 DAgger 的核心动机一致，但没有累积并重放历史数据集。

评估使用三道防线：未参与训练的固定 probe；相同随机开局和执子的 pre/post 配对对局；以及完全禁用学习的身份对照。

## 4. 实验结果

### 4.1 原生 5×5 Sayuri-policy

每个条件使用 3 个 seed、4 盘在线训练、2 盘 pre/post 配对评估和 16 个固定 probe。表中为 seed 均值 ± 样本标准差，正的 loss 改善表示 loss 下降。

| 条件 | probe loss 改善 | raw 合法率变化 | masked 一致率变化 | 实战目差改善 |
|---|---:|---:|---:|---:|
| rules + MQR + LoRA | +0.335±0.067 | +27.1±9.5 pp | +20.8±13.0 pp | **-11.5±8.2** |
| rules + no learning | 0 | 0 | 0 | 0 |
| board-only + MQR + LoRA | +0.368±0.107 | +33.3±9.5 pp | +22.9±9.5 pp | -14.0±8.8 |
| rules + MQR only | +0.335±0.067 | +27.1±9.5 pp | +20.8±13.0 pp | -11.5±8.2 |

rules-online 的评估教师一致率平均提升 28.5 个百分点，但 6 个配对对局的目差变化为 `-3,-16,-25,-16,0,-9`。这是“在线权重学习有效”与“解决实际任务无效”同时成立的直接证据。

失败机制是可观测的 episodic recency。例如 seed 2026 的最后四分之一流中，13/13 次 raw 预测都是 `pass`。每局结尾总是 pass 标签，最近更新将下一局开局也推向 pass，学生轨迹因而自我加强早停。低秩 OGD 并未消除这一类功能漂移。

单 seed 的 7×7 原生教师复验也从 47–52 手训练对局缩短到 11–14 手，配对目差变化 -26。因此不能将失败完全归因于 5×5 棋盘，但该 7×7 结果只有一个 seed。

### 4.2 pass 课程与独立评估

优化条件在占用率 60% 前从 Sayuri raw policy 候选中排除 pass，并令 pass 标签仍计入在线 loss，但暂时不提交权重更新。这是一个显式课程干预，不是原生 Sayuri 结果。

| 条件 | probe loss 改善 | raw 合法率变化 | 课程对手目差改善 | 原生 Sayuri 目差改善 |
|---|---:|---:|---:|---:|
| rules + MQR + LoRA | -0.046±0.196 | -31.3±28.6 pp | **+6.83±7.29** | **+6.50±7.70** |
| rules + no learning | 0 | 0 | 0 | 0 |
| board-only + MQR + LoRA | -0.120±0.128 | -14.6±13.0 pp | +1.67±9.28 | +1.50±8.19 |
| rules + MQR only | -0.046±0.196 | -31.3±28.6 pp | +6.83±7.29 | 未重复；其权重轨迹与启用 LoRA 时一致 |

课程内的 6 个 rules 配对目差都不为负：`+6,+3,+2,0,+16,+14`。但种子是正确的统计单位，n=3 时均值的 95% t 区间为 `[-11.27,+24.93]`；不能把 6 盘棋当作 6 个独立模型。

更重要的是，目差改善伴随 raw 合法率下降，而且固定 probe loss 平均略微变差。这说明课程改变了闭环行为并可能改善得分，却没有学出可移除外部 mask 的规则策略。

### 4.3 规则文本是否被语义使用

在相同课程训练、原生 Sayuri 交叉评估中：

| prompt | 目差改善 | probe loss 改善 | raw 合法率变化 |
|---|---:|---:|---:|
| 正确规则 | +6.50±7.70 | -0.094±0.243 | -10.4±23.7 pp |
| 错误规则 | +5.33±7.15 | -0.017±0.148 | -4.2±20.1 pp |
| board-only | +1.50±8.19 | -0.106±0.102 | -4.2±9.5 pp |

正确规则与错误规则的目差改善差值是 `-2.5,+2.5,+3.5`，均值 `+1.17±3.21`。95% 区间跨 0。这个负对照否定了当前数据上的强语义解释。

## 5. 数学上能证明什么

### 5.1 常量规则 prompt 的不可识别性

设所有训练样本都使用同一规则串 \(r_0\)，数据为 \((r_0,b_t,y_t)\)。任意在观测切片上拟合数据的函数 \(g(r_0,b)\)，都存在一个完全忽略规则的函数 \(\bar g(b)=g(r_0,b)\)，两者的经验风险完全相同。因此，只从固定正确 prompt 上的风险下降，无法识别模型是否使用了规则语义。

如果冻结编码器还可分解为

\[
f(r,b)=u(r)+v(b),
\]

则对线性下游头 \(Wf=Wu(r_0)+Wv(b)\)，规则部分只是可吸收到偏置的常量。规则只可能通过预训练注意力的 \(r\times b\) 非线性交互、LoRA 表示变化或非线性 MQR 影响局面间的相对映射。正确/错误规则的反事实消融是必要条件，而本次没有通过该条件。

### 5.2 环的稳定性与“记忆”边界

MQR 以反 Hermitian \(A\) 构造 Cayley 变换 \(U=(I-A)(I+A)^{-1}\)。由 \(A^\dagger=-A\) 可得 \(U^\dagger U=I\)；因此 \(H_{ij}=|U_{ij}|^2\) 非负且行、列和均为 1。对 1-Lipschitz 的 tanh，

\[
T(h)=\tanh((1-\alpha)hH_\beta^\top+\alpha J(z))
\]

在 \(L_\infty\) 下的压缩率至多是 \(q=1-\alpha\)，因此存在唯一不动点。当前 \(\alpha=0.3,K=24\)，上一局面状态对充分松弛结果的影响上界为

\[
q^K=0.7^{24}\approx1.92\times10^{-4}.
\]

所以当前行为的持久变化主要是**参数学习**，不是长时间的环激活记忆。若要模拟快/慢动物记忆，必须使用少量单步更新或独立慢环，而不能同时充分求平衡又宣称长时状态保留。

### 5.3 正交更新的精确含义

设分块学习率预条件矩阵为 \(D\)，历史白化梯度的正交基为 \(Q\)，\(P=I-QQ^\top\)。实现的一阶方向是

\[
\Delta\theta=-D^{1/2}PD^{1/2}g.
\]

因此

\[
g^\top\Delta\theta=-\|PD^{1/2}g\|_2^2\le0,
\]

而对被 \(Q\) 完整张成的旧梯度 \(g_o\)，\(g_o^\top\Delta\theta=0\)。若损失是 \(L\)-smooth，只能再得到

\[
\ell(\theta+\Delta)-\ell(\theta)
\le -\|PD^{1/2}g\|^2+\frac L2\|\Delta\|^2.
\]

这是单步局部条件，不是长期不遗忘定理。历史基只有 rank 8，旧梯度会变陈旧，二阶项和未记忆方向仍可以造成强烈功能漂移。pass 塌缩正是反例。

“整体正交参数更新”也容易误导：Cayley 只保证内部 \(U\) 保持酉；OGD 只保证更新向量对有限梯度子空间正交。注入、读出和 LoRA 参数本身并不是正交矩阵。

### 5.4 为什么模仿 loss 不保证目差

固定特征、凸线性头、有界梯度和投影域下，普通 OGD 有

\[
\operatorname{Regret}_T
\le \frac{D_\Theta^2}{2\eta}+\frac{\eta G^2T}{2},
\]

取 \(\eta=D_\Theta/(G\sqrt T)\) 可得 \(O(\sqrt T)\) 静态 regret。当前 MQR 与 LoRA 联合是非凸且表示在变，因此该界不直接适用。

即使模仿 regret 很小，终局收益也没有同号保证。性能差异可写成教师 advantage 在学生访问分布上的累加：

\[
J(\pi)-J(\pi_E)
=\sum_t\mathbb E_{s\sim d_\pi^t}
[A^{\pi_E}(s,\pi(s))].
\]

交叉熵只控制动作不一致概率，不控制每次不一致的 advantage 大小，更不能修复教师的过早 pass。学生轨迹上取标签可减少离线行为克隆的分布偏移，但不能消除代理目标错配。

## 6. 下一版算法：从根本上拆分三个问题

当前 26 类单头把“是否结束”、“哪里合法”和“合法着中哪个更好”混在一起。建议改为分层策略：

\[
\pi(a\mid s)=
\begin{cases}
g_{pass}(s), & a=\text{pass},\\
(1-g_{pass}(s))\,\pi_{point}(a\mid s), & a\in\{1,\ldots,B^2\}.
\end{cases}
\]

1. **合法性头** \(\hat m(s)\in[0,1]^{B^2}\)：直接用环境生成的全合法 mask 做多标签训练，独立报告 occupied、suicide、ko 错误。只有 raw 合法率达到门槛后，才可在研究评估中移除外部 mask。
2. **落点策略头** \(\pi_{point}\)：在合法非 pass 点上学 Sayuri 软策略，避免 hard argmax 丢失相对概率。
3. **终局头** \(g_{pass}\)：使用类别平衡的二元损失，输入占用率、上一手是否 pass 和价值估计，不再让每局最后两个标签覆盖全部落点读出。
4. **价值头** \(V(s)\)：用最终目差或 TD 目标训练，使学习目标与实际得分直接对齐。
5. **分层回放与慢环巩固**：按开局/中盘/终局、pass/非 pass、黑/白子分层的小型 reservoir 解决时序偏置；OGD 在局间巩固时更新，不代替回放。

联合目标可写为

\[
\mathcal L=
\mathcal L_{point}
+\lambda_p\mathcal L_{pass}
+\lambda_m\operatorname{BCE}(\hat m,m)
+\lambda_v(V(s)-G)^2.
\]

规则学习也必须改成可识别设计：在训练中随机切换自杀、ko、计分和终局规则，并要求模型根据 prompt 预测不同合法 mask/策略。固定一条规则永远无法识别规则文本的因果作用。

## 7. 推广到 2B 与“动物学习”

工程上，88,320 个 MQR+LoRA 参数相对当前冻结主干约为 0.01%，外挂方案可以扩到 2B。但当前 checkpoint 是约 0.88B 个已加载主干参数，并非 2B；而且执行时是冻结 FP16，所以 2B 会近似翻倍主干显存，不能用 AWQ 文件大小估算运行开销。

科学上还缺少：冻结 LM head 的 token 级验证、延迟反馈队列、KV-cache/环状态对齐、主动探索、世界模型、稀疏奖励归因、离线回放巩固和无任务 ID 路由。当前实验最多对应“教师监督下的快速联想适应”，不对应一般动物学习。

## 8. 下一阶段的可证伪门槛

- **权重学习有效**：相对 no-learning，独立 probe 或独立对手上的指标需跨 seed 稳定改善；参数漂移本身不够。
- **学会规则**：正确规则必须显著优于错误规则和 board-only，并在无 mask 时准确处理 occupied/suicide/ko。
- **LoRA 有价值**：与 MQR-only 的行为差异必须超过多 seed 噪声和运行开销。
- **解决实际问题**：至少 20 个 seed、更多配对开局、独立教师/对手和预注册主指标；同时报告胜率、目差、raw 合法率、忘却、时延和内存。

## 9. 复现

```bash
python3 test_go_minicpm.py
python3 analysis/minicpm_go_real_games_v2_verify.py

# 原生 Sayuri：替代损失改善、实战目差退化
python3 experiments/minicpm_go_real_games.py \
  --seeds 2026 2027 2028 \
  --conditions rules-online rules-no-learning board-only-online rules-mqr-only \
  --train-games 4 --eval-games 2 --probe-positions 16 \
  --metrics-path analysis/results/minicpm_go_real_games_3seed.json

# pass 课程 + 原生 Sayuri 交叉评估 + prompt 负对照
python3 experiments/minicpm_go_real_games.py \
  --seeds 2026 2027 2028 \
  --conditions rules-online wrong-rules-online board-only-online \
  --train-games 4 --eval-games 2 --probe-positions 16 \
  --min-pass-occupancy 0.6 --pass-update-scale 0 \
  --evaluation-teacher native-policy \
  --metrics-path analysis/results/minicpm_go_real_games_prompt_controls_3seed.json
```

完整原始数据还保留每一个训练手的更新强度、原始/掩码动作、教师标签、OGD 秩和因果顺序审计位。
