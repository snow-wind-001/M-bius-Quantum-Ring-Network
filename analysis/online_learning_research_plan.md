# 环状在线学习研究方案与 2B 推广路线

## 当前实现边界

本轮已经实现研究级向量模型与 MiniCPM5-1B 隐状态桥接，而不是完整的 2B 生成式适配器：

- `OnlineMultiRingClassifier`：显式上下文到环的稳定 Top-1 路由、每环独立状态与参数、容量耗尽默认报错；
- prequential 原子协议：先用 \(\theta_t\) 预测，再用当前反馈更新到 \(\theta_{t+1}\)，返回值始终是更新前预测；
- `OrthogonalGradientMemory`：对读出、注入、Cayley 实/虚坐标及可选原型组成的完整梯度做一次统一投影；
- 阶段化巩固：当前上下文内可只收集历史方向、不启用投影，切换上下文后再保护；
- 状态和梯度基均进入 `state_dict`；v6 已增加无任务 ID 的过去波动硬路由，
  但它仍是固定阈值的可识别机制实验，不是已学得的通用路由器。
- `UtilityDrivenMQR` 用配对影子轨迹学习未来写入收益；v6 分专家门在
  正确硬路由下给出未选专家精确零梯度/零漂移。
- `mqr/minicpm.py` 将本地 AWQ INT4 checkpoint 一次解包为冻结 FP16 主干，在末层 `q_proj/v_proj` 加 FP32 LoRA，并接受 MQR 的外部特征梯度；当前输出仍是独立围棋动作头，不经过语言模型 LM head。
- `mqr/sayuri.py` 通过独立 GTP 子进程接入 Sayuri；5×5 规则实现和引擎在抽查局面上合法着一致。

推理与学习的“同时”在这里指同一个在线事务中的 read-before-write，而不是两个线程并发改写参数。后者会产生竞态，也破坏可复现的 \(\theta_t\to\theta_{t+1}\) 语义。

## 严格更新规则

设第 \(i\) 个参数块的实际步长为 \(\eta_i\)，令

\[
D=\operatorname{diag}(\eta_i I_i),\qquad w_t=D^{1/2}g_t.
\]

历史白化梯度的正交基为 \(Q\)，则实现采用

\[
w_t^\perp=(I-QQ^\top)w_t,
\qquad
\Delta\theta_t=-D^{1/2}w_t^\perp.
\]

对任意满足 \(D^{1/2}g_{old}\in\operatorname{span}(Q)\) 的旧梯度，

\[
g_{old}^\top\Delta\theta_t
=-(D^{1/2}g_{old})^\top w_t^\perp=0.
\]

当前损失的一阶变化则为

\[
g_t^\top\Delta\theta_t=-\|w_t^\perp\|_2^2\le0.
\]

这解释了为什么不同参数块不能先后独立更新，也不能直接在未白化梯度上投影。保证要求参数布局和相对学习率保持不变。若旧损失梯度为 \(L\)-Lipschitz，只能推出

\[
L_{old}(\theta+\Delta\theta)-L_{old}(\theta)
\le \frac{L}{2}\|\Delta\theta\|_2^2,
\]

因此它是单步局部保证，不是长期零遗忘定理。历史方向会随参数移动而变旧；低秩基、二阶项和多步累积都可能造成遗忘。

显式多环给出更强但有条件的结论：若上下文 A 永远路由到环 A，学习 B 时只修改环 B，且共享主干与路由被冻结，则 A 的参数、状态转移和无状态输出严格不变。自动新颖性路由发生误选时，这一保证立即消失。

## Digits 初步原理验证

实验使用 `sklearn.datasets.load_digits`，无需下载。训练集只流过一次；A 为标准化 64 维像素，B 为同一数据的固定带符号特征置换。每个 minibatch 先评分再更新。结果为 3 个种子均值 ± 总体标准差：

| 方法 | 可训练参数 | A 学完 | B 后 A | A 遗忘 | B 学完 | B prequential |
|---|---:|---:|---:|---:|---:|---:|
| 共享单环 | 6,880 | 83.33±1.20% | 65.37±3.64% | 17.96±2.92 pp | 89.38±0.53% | 76.05±1.24% |
| 共享单环 + OGD-64 | 6,880 | 83.33±1.20% | 68.70±2.52% | 14.63±3.32 pp | 88.21±0.89% | 74.91±1.05% |
| 显式隔离双环 | 13,760 | 83.33±1.20% | 83.33±1.20% | **0.00±0.00 pp** | 85.25±0.53% | 64.57±1.43% |

OGD 把平均遗忘降低约 18.6%，但三个种子中有一个变差，同时 B 最终准确率下降约 1.17 pp，体现稳定性—可塑性权衡。投影后对白化历史子空间的最大内积约 \(1.9\times10^{-8}\)，说明代数约束成立；这并不等于测试集函数保持。双环对 A 的参数和 logits 漂移均为精确 0，但参数翻倍，且失去共享单环对 B 的迁移收益。

OGD-64 的稠密基占 1,761,280 字节，约为模型参数存储的 64 倍；它适合验证定理，不适合直接复制到大量 2B sidecar。完整配置和逐次结果见 `analysis/results/online_digits_principle.json`。

## Temporal MQR-v2 状态记忆门槛

固定点环在 `alpha=0.3,K=24` 时每个外部观察只保留
`0.7^24≈1.92e-4` 的旧状态。`mqr/temporal.py` 因此改为每个观察只推进
一次，并将遗忘率 `lambda` 与写入强度 `kappa` 分离。3-seed 延迟 Digits
中，equilibrium 控制为 `9.06±1.43%`，单慢环为 `81.04±1.00%`，
多尺度 unistochastic 为 `77.50±1.13%`；同初始化 no-learning 为
`8.96±3.08%`。`kappa=1` 相对 `kappa=lambda` 改善
`55.31±7.10 pp`，说明慢环必须能够强写。

该结果没有证明环拓扑优势：逐位同初始化的 unistochastic 比 identity 低
`1.25±0.83 pp`，固定总状态预算下多尺度比单慢环低 `3.54±0.18 pp`。

后续 v3 已加入真实 digit 干扰与外部受控写门。多尺度 oracle cue-only 门为
`73.44±5.16%`，无门为 `20.10±1.83%`，配对改善
`+53.33±6.17 pp`；同稀疏度随机门仅 `18.13±1.13%`。这证明“正确选择写入
时机”能保护状态，不证明模型自主学会门。oracle 下多尺度仍比单慢环低
`2.81±4.33 pp`，继续没有多尺度优势证据。运行时同时增加了容量安全的 keyed
state bank，以及缓存旧 logits/读出向量的单次延迟反馈 ticket。严格公式、
结果和限制见 `analysis/temporal_mqr_runtime_report.md`。

v4 已把外部 oracle 缺口缩小为**即时辅助事件监督**：低秩门先用旧参数决定
当前写入，随后 cue/non-cue 标签才能更新独立门参数；任务梯度在门处严格
detach。门有独立 OGD、范数裁剪、计数和 checkpoint，外部门可覆盖实际写入
并在后台训练控制器。27 项 Temporal 测试验证当前标签不能改变当前 gate、state
或 logits。

3-seed marker-Digits 中，44 参数多尺度学习门为 `70.83±6.55%`，无门为
`19.86±1.68%`，不更新门为 `21.53±0.87%`；learned 相对 ungated
配对改善 `+50.97±6.74 pp`，oracle 只再高 `1.94±1.58 pp`。控制器只读取
显式非类别 marker，并在每帧后收到即时标签，因此这是“事件识别可在线学习”
的原理验证，不是稀疏奖励信用分配。10%/25% false-open 分别造成
`20.28±5.37`/`33.75±7.23 pp` 退化，约 10% false-close 造成
`7.36±0.87 pp` 退化；写门的 false-open 是首要风险。单慢环仍比多尺度高
`2.78±1.20 pp`。

## Temporal MQR-v5/v6 无 marker 收益与反转门槛

v5 已将目标从“识别事件”改为回归一次写入的未来反事实收益。
3-seed marker-free Digits 中，OGD 收益门为 `72.00±6.24%`，但目标仍可由
新颖性直接识别。v6 进一步用严格过去的波动迹分开反转上下文，没有任务位或
显式 marker。

5 个新 seed 的 A→B→A 流在 288–320 可塑参数和 4096 B 在线张量状态上限
内比较了 GRU、fast-weight、LoRA、OGD-LoRA 和 equal-byte replay。双专家
MQR+OGD 的 A-after-B 下降为 `0.0 pp`，相对 LoRA 减少 `36.75 pp`，
bootstrap 95% CI `[34.00,39.50]`；最终平衡准确率高 `11.63 pp`。
但 LoRA 的全过程平均阶段准确率高 `8.58 pp`，延迟约低 75 倍，且总
sidecar 参数仅 300，而 MQR 为 1354。这建立了可分上下文中的独立稳定性
机制，没有建立等总资源的整体优势。下一步必须学习路由、匹配总参数/FLOP，
并将精确影子轨迹替换为经校准的 critic 或 Jacobian sketch。

## 推广到冻结 2B 主干

推荐只在最后隐藏状态处先做外挂：

\[
z_t=\operatorname{stopgrad}(f_{2B}(x_{\le t})),\quad
\hat z_t=z_t+\gamma_k Q_k^{out}\,h_{k,t},\quad
\ell_t=E^\top\operatorname{Norm}(\hat z_t).
\]

第一版取 \(N=64/128\)、rank 8、\(K=1\sim3\)、Top-1 环，只在线更新输入/输出低秩投影；Cayley 坐标每 32–256 个样本慢速更新并缓存 \(U/H\)。不能每 token 重算稠密 Cayley 解。

MiniCPM5 机制实验已经补上“冻结主干隐藏状态 → MQR 动作损失 → 外部 LoRA VJP”这段梯度链。Temporal v4 已解决 sidecar 内部会话状态生命周期、辅助事件门和读出级延迟反馈，但距离生成式 2B 接口仍有三个实质缺口：

1. 接受冻结 LM head 传回的 next-token 状态梯度，而不是只在环的 26 类动作 logits 上计算交叉熵；
2. 将现有 `stream_id`/ticket 与 token、KV-cache、模型版本原子对齐，并为延迟
   LoRA/注入更新提供参数快照或可复现 feature replay；
3. 用随机投影、分块基或特征/Jacobian sketch 代替 \(O(mP)\) 的稠密 OGD 存储。

多局围棋实验还表明，动作头需要拆分为合法性、非 pass 落点、pass 门和价值头。
v2 五种子中，掩码后目差改善 `+5.75`，但 95% t 区间为
`[-2.25,+13.75]`；原生 probe loss 与 pass Brier 在五个 seed 上全部恶化。
详见 [`mqr_reversal_go_v2_report.md`](mqr_reversal_go_v2_report.md)。

## MiniCPM5–Sayuri 初步门槛结果

在 RTX 3070 Laptop 上，冻结主干 + 两个 MQR + rank-8 LoRA 的峰值 CUDA 分配约 2.20 GB，在线单步约 38 ms。Sayuri-policy 24 步单盘流完成一局，held-out loss 为 `3.289→3.170`；8 个基础局面循环 48 步时，首/末四分位 loss 为 `3.149→2.249`，回放 masked 教师一致率为 `75%`。

严格 no-LoRA 对照得到几乎相同指标，逐步 loss 平均绝对差只有 `2.17e-5`。所以该门槛只证明 1B 主干、MQR 和 LoRA 的在线链路可运行，不能证明 LoRA 增益或外推到 2B。完整设置、数学链式法则与限制见 `analysis/minicpm_sayuri_online_report.md`。

## 分阶段基础研究

1. **已完成：机制闭环。** 数值梯度、prequential 无泄漏、状态续接、多环零漂移、OGD 等式、Digits A→B，以及 Temporal MQR 的延迟状态记忆与写入/遗忘消融。
2. **已完成：首轮实用运行时。** 外部写门、真实 digit 干扰、keyed state bank、
   有界单次 feedback ticket、LRU/过期/恢复测试；证据支持选择性写入。
3. **已完成：辅助学习门门槛。** 因果低秩门、独立 OGD/裁剪/恢复、门控腐败
   曲线和 marker-Digits 已验证；证据支持即时事件监督，不支持自主门控或
   多尺度优势。
4. **已完成第一轮 marker-free 反转压测。** 过去波动硬路由已验证专家隔离，
   但它依赖人工选定阈值与可分上下文；乱序反馈、不可分反转和学习路由尚未完成。
5. **已完成同可塑参数/状态上限基准；继续同总资源基准。** 已比较 GRU、
   fast-weight、LoRA、OGD-LoRA 和 replay；下一轮必须同时匹配总 sidecar
   参数、FLOP 和延迟预算，再进入 Split/Permuted MNIST 与 Split CIFAR-100。
6. **学习无任务 ID 路由。** 将固定波动阈值替换为带拒识的 change-point posterior
   或在线聚类；分别报告路由准确率、条件于正确路由的遗忘和路由误差。
7. **小语言模型门槛。** 隐状态分类桥接已在约 1B 主干上跑通；下一门槛是冻结 LM head 的一次性事实修正、风格切换和奖励反转。只有在等预算下优于 LoRA/replay 才扩到 2B。
8. **2B 验证。** 报告 next-token loss、旧域退化、每 token P50/P95 延迟、峰值显存、每环容量、ticket staleness 和长期路由漂移。

与“动物学习”相关的可证伪子任务应拆成一次性联想、上下文切换、延迟匹配、奖励反转和离线回放巩固。主动探索、世界模型、自监督预测、稀疏信用分配和身体闭环尚不存在，因此当前系统只能称为受控的快速持续适配器，不能称为一般动物学习模型。
