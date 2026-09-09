# MQR 机制、容量与独立优势审计

> 状态：2026-09-01 正式证据。本报告保留经独立验证器通过的
> schema-v2 机制实验、等状态容量扫描和 10×10/13×13 围棋正式结果，并追加
> 内容/地址分离路由的 10-seed 资格结果。
> Smoke/pilot 不用于性能结论。

## 1. 结论

MQR 的数学构造已经自洽，在线信用链也真实连通，但当前不具备独立
算法优势。这不是 Cayley 坐标漏掉了整族酉方向导致的：历史参数化
完整张成 \(\mathfrak u(N)\)，而且全局相位可使 Cayley--模平方复合映射
覆盖全部 unistochastic 转移。真正的瓶颈是：

1. \(U\mapsto |U|^2\) 丢弃相位，局部可见秩最多为 \((N-1)^2\)，在
   identity/置换点一阶秩为 0；
2. 双随机混合是不扩张约束，不是信息保真定理。在当前任务中，identity
   比环混合更好地保留了隐变量；
3. 有限轨迹重放给出精确的有限 BPTT 梯度，但收缩只给信用范数
   上界，不给任何正下界；
4. 自主 write critic 几乎把事件和干扰都写入，尚未学会“保存是否
   改善未来行为”；
5. 结构刷新可用慢时标摊薄，但单次 Cayley 刷新的峰值更新成本
   仍明显超过 identity。

因此，当前合理定位是“有明确结构约束和安全事务的在线 sidecar
研究原型”，而不是围棋学习器、通用小模型增强器或动物式学习系统。

2026-09-01 的地址路由改造解决了第 2、4 项在合成任务上的直接症状：内容不再被
双随机矩阵反复平均，预算化门能分开事件和干扰。它通过了路由机制门，但与直接
固定环形缓冲区严格持平，因此没有推翻“独立算法优势未成立”的总判断。

## 2. 本轮修正

### 数学与代码

- `CayleyUnistochasticParam.set_from_unitary_representative_()` 对多于 \(N\)
  个全局相位计算 \(\sigma_{\min}(I+\zeta U)\)，选择条件最好的有限逆
  Cayley 坐标。固定 float64 例子的 \(H\) 重建误差为
  `4.44e-16`。
- 补充有限轨迹信用定理：重放已提交的同版本轨迹可得到该有限图
  的精确梯度；\(\|\partial h_t/\partial h_s\|_2\le(1-\lambda)^{t-s}\)
  仅为上界。数值检查同时验证了收缩上界和 tanh 饱和时的零梯度反例。
- Agent 反馈返回 `transition_refresh_required`，仅当事务真正提交、
  慢 tick 到期且活跃块存在 Cayley 参数时为真。容量实验因而计数
  真实刷新，而非全局日程 tick。
- Sinkhorn 的 full-leak 环没有 recurrent carry，其转移从未进入前向。实现将
  该未用参数改为保留历史 state-dict key 的 buffer；它仍消耗同一个 seeded
  随机抽样以保持后续初始化可复现，但不再被错计为 Sinkhorn 独有的冻结
  参数。这使九方法冻结基底数量恢复精确一致，没有使用 dummy padding。

### 实验语义

- 机制实验 schema 升为 2，三个重复 query 均计入 query，不再将其中
  两个错计为干扰。
- seed 区间统一为双侧 95% Student-\(t\) 区间；验证器从原始 seed
  重算区间、配对差、查询计数、资源比和每个决策门。
- “独立优势”现在同时要求：击败 identity 和全部非 MQR 核心、反转
  学习为正、遗忘更少、未来信用达到注入/转移、门不坍缩且资源门
  通过。任一项失败都必须返回 false。

## 3. 10-seed 机制资格实验

任务是长度 12 的 marker-free 延迟记忆序列。首个观测包含一个隐比特，
其后有干扰和三个输入完全相同的 query。所有预测先提交，轨迹结束后
才做一次更新。每 seed 使用 160 个主训练 episode、40 个反转 episode、
40 个 critic 校准 episode、40 个普通评估和 40 对平衡历史评估。

| 方法 | 历史准确率 | 历史概率分离 | B 反转准确率 | A 遗忘 |
|---|---:|---:|---:|---:|
| cyclic MQR | 0.5375 `[0.5119,0.5631]` | 0.01178 `[0.00671,0.01685]` | 0.3925 `[0.3219,0.4631]` | -0.0625 `[-0.1719,0.0469]` |
| identity | 0.5875 `[0.5493,0.6257]` | 0.02946 `[0.01574,0.04318]` | 0.3700 `[0.2841,0.4559]` | -0.0100 `[-0.0685,0.0485]` |
| GRU | 0.5000 `[0.5000,0.5000]` | -0.00001 `[-0.00005,0.00002]` | 0.4675 `[0.4307,0.5043]` | -0.0175 `[-0.1091,0.0741]` |
| LSTM | 0.5000 `[0.5000,0.5000]` | 0.00000 `[-0.00000,0.00001]` | 0.4925 `[0.4245,0.5605]` | -0.0250 `[-0.1174,0.0674]` |
| fast-weight | 0.5000 `[0.5000,0.5000]` | 0.00023 `[-0.00006,0.00052]` | 0.4950 `[0.4514,0.5386]` | 0.0100 `[-0.0631,0.0831]` |
| Sinkhorn | 0.5100 `[0.4874,0.5326]` | 0.00367 `[-0.00165,0.00899]` | 0.4800 `[0.4372,0.5228]` | 0.0200 `[-0.0629,0.1029]` |

最关键的配对差是 cyclic-minus-identity `-0.0500`，95% CI
`[-0.07890,-0.02110]`，即 identity 显著更好。MQR 相对 GRU/LSTM/fast-weight
的正差不能建立总优势：它已输给最直接的拓扑消融 identity，对
Sinkhorn 的区间也跨 0，且六种核心的总参数/MAC 资源门未匹配。

信用连通性确实成立：未来损失到注入、转移和最早输入的平均梯度范数
分别为 `0.19594 [0.15480,0.23709]`、`0.02376 [0.01922,0.02829]`
和 `0.03165 [0.02109,0.04221]`。但将 MQR 转移替换为 identity 的准确率降幅
仅 `0.0025 [-0.0417,0.0467]`，未显示环转移的行为必要性。

write critic 的平均总写率为 `0.8184`，事件写率为 `1.0000`，干扰
写率仍为 `0.8934`。这是“门未坦缩成全关”，不是“门已学会选择”。
反转增益为 `-0.0675 [-0.1493,0.0143]`，也没有学会自主改写规则。

## 4. 等状态容量扫描

容量扫描只比较 cyclic MQR 和 identity，两者共享数据、路由、头、损失、
反馈和完全相同的循环状态字节。环宽对应 48/192/768 B，Cayley 参数
每 32 次更新刷新一次；每 seed 实测恰好 5 次，identity 为 0 次。

| 状态 | MQR 历史准确率 | identity 历史准确率 | 配对差 MQR-id | 参数比 | 摊销更新比 | 峰值更新比 |
|---:|---:|---:|---:|---:|---:|---:|
| 48 B | 0.5200 `[0.4778,0.5622]` | 0.5488 `[0.5013,0.5962]` | -0.02875 `[-0.05766,0.00016]` | 1.0167 | 1.0050 | 1.1596 |
| 192 B | 0.5438 `[0.5051,0.5824]` | 0.5763 `[0.5252,0.6273]` | -0.03250 `[-0.07511,0.01011]` | 1.0303 | 1.0223 | 1.7122 |
| 768 B | 0.5775 `[0.5352,0.6198]` | 0.6488 `[0.5981,0.6994]` | -0.07125 `[-0.13132,-0.01118]` | 1.0381 | 1.0410 | 2.3110 |

三档状态的参数、状态、前向和摊销更新比都通过 1.05 门，所以“不过
是 MQR 容量太小”不能解释败因。状态变大时，MQR 和 identity 都提高，
但 identity 提高更多；768 B 上已出现显著的 MQR 劣势。三档峰值更新
门都失败，且环越大越严重。因此
`mqr_capacity_advantage=false`，
`mqr_peak_qualified_capacity_advantage=false`。

## 5. 与 10×10/13×13 围棋证据的交叉结论

围棋正式实验已经解决“棋盘太小”和“基线/状态资源不公平”两个
旧问题。10×10/13×13 各用 10 seeds，九种方法的参数、状态接口、前向和
更新分配比均低于 1.05。MQR 的 task-A held-out loss 分别下降
`2.8331 [2.7121,2.9531]` 和 `3.0168 [2.9378,3.0929]`，证明权重
在真正在线更新。

然而，两尺度、所有方法的历史焦点都是 `0.500`，且任务 A 正遗忘全部
为 0。MQR 与 identity 及多个递归对照基本同轨；13×13 上 replay-LoRA
的动态有效合法率和回报还显著更好，MQR-minus-replay 分别为
`-0.00407 [-0.00633,-0.00136]` 和
`-0.1125 [-0.1750,-0.0375]`。结合容量扫描可以排除两个简单解释：

- 失败不只是 3×3 棋盘或 5 seeds 造成的统计功效不足；
- 失败不只是状态字节不公平或环容量太小。

更可能的原因是当前任务与目标没有使“环混合”成为有用且可识别的
计算，而写门又没有自主学出可泛化的保存效用。

## 6. 根因分层

### 不是致命代数错误

当前 Cayley 坐标、非线性伴随、模平方与 Cayley 拉回都经过 float64
对拍。因而不应把负结果简单归因于“数学公式仍然写错”。已修正的旧
问题包括非线性 \(D\ne I\) 伴随、\(dH\) 一阶项、有限求解认证以及
历史错误 Cayley 更新。

### 是结构性表示与优化限制

- 线性平衡态严格等价于低秩线性适配器；非平凡收益必须来自时序、
  非线性或选择写入。
- \(H=|U|^2\) 保留非负与双随机结构，却丢弃能使 \(U\) 保范的相位。
  当任务只需直接保留一个比特时，identity 是更强的归纳偏置。
- 非平坦初始化已使 transition gradient 非零，但“有梯度”不等于“梯度
  改善未来决策”。模平方不可见方向、收缩和损失相消都会降低
  功能信用。
- 当前 critic 的双轨迹目标在因果上正确，但数据分布与自举过程使其
  学成近似常开门。这是估计/探索算法的失败，不是门接口不连通。

### 是任务识别性不足

围棋正式协议中的历史依赖仍太短，空间支路与共享头可以主导大部分
损失；任务 B 也没有造成可测的 A 遗忘。所以即使系统在学习，环与 OGD
也没有被任务强制贡献可识别的性能。

## 7. 后续修改顺序与停止门

1. **先修写门，不先扩棋盘**：平衡收集正/负未来收益票据，报告
   advantage 校准、AUPRC、event/distractor 写率和密度约束。未达到
   显著 event--distractor 间隔时，停止 Go/LLM 放大。
2. **让环成为可识别计算**：在不使用 marker 的情况下，预注册变长延迟、
   多次查询、地址置换和上下文反转任务。需要同时超过 identity 和至少
   一个非 MQR 时序核心，且 identity replacement 必须显著降低性能。
3. **做功能而非坐标对比**：加入转移冻结、identity、随机重排、只改相位、
   \(H\)-打乱和读出重拟合消融，分开“参数变了”与“环计算被使用”。
4. **控制峰值成本**：优先测试 \(O(N)\)/\(O(N^2)\) 的局部 Givens
   刷新、分块 Cayley 或仅当转移收益过门时刷新。摊销门和 P95/峰值门
   必须同时报告，不得用前者覆盖后者。
5. **保留安全 sidecar 定位**：在冻结小模型上只用小子空间
   \(N=32\ldots128\)、零初始 residual、慢转移和原子 rollback。在上述机制
   门通过前，不启用长轨迹策略 RL 或把结果外推到 2B 通用能力。

## 8. 学术与应用价值

当前成果有学术价值，但价值类型是可证伪的机制与负结果研究：它给出
正确的非线性隐式梯度、Cayley--unistochastic 表示分析、有限轨迹信用边界、
可回滚的在线事务，并用预注册失败门排除了“小棋盘”、“seed 太少”和
“状态不公平”等简单解释。如果按算法性能论文定位，当前证据不足；
如果定位为可审计在线记忆 sidecar 的机制报告或 workshop 工作，则结论
是完整且诚实的。

应用上，它可继续用于冻结小模型的有界会话状态、受控个性化和错误
修正，因为这些场景直接需要零初始扰动、按流状态、资源上界和回滚。
但环核心在这些应用中是否优于简单 identity/replay 仍必须逐项消融，不能
由安全性直接推出。

## 9. 复现

```bash
python3 experiments/mqr_mechanism_qualification.py \
  --seeds 10 --seed-start 41 --ring-dim 24 --sequence-length 12 \
  --train-episodes 160 --reversal-episodes 40 \
  --oracle-warmup-episodes 100 --critic-calibration-episodes 40 \
  --eval-episodes 40 --paired-episodes 40 \
  --transition-update-interval 2 --workers 4 \
  --output analysis/results/mqr_mechanism_qualification_formal_10seed.json
python3 analysis/mqr_mechanism_qualification_verify.py \
  analysis/results/mqr_mechanism_qualification_formal_10seed.json

python3 experiments/mqr_cyclic_capacity_sweep.py \
  --seeds 10 --seed-start 41 --state-bytes 48 192 768 \
  --sequence-length 12 --train-episodes 160 \
  --oracle-warmup-episodes 100 --critic-calibration-episodes 40 \
  --paired-episodes 40 --transition-update-interval 32 --workers 4 \
  --output analysis/results/mqr_cyclic_capacity_sweep_formal_10seed.json
python3 analysis/mqr_capacity_sweep_verify.py \
  analysis/results/mqr_cyclic_capacity_sweep_formal_10seed.json
```

原始结果为
`analysis/results/mqr_mechanism_qualification_formal_10seed.json` 和
`analysis/results/mqr_cyclic_capacity_sweep_formal_10seed.json`。

## 10. 内容/地址分离路由复验

新版状态写成物理内容 (C_t) 与逻辑地址 (a_t)。无写事务时
(C_{t+1}=C_t) 精确成立；残差双随机算子只生成地址分数，随后以
straight-through top-1 提交离散地址。这样消除了旧版“为路由而反复平均内容”的
结构冲突。写门同时加入逐前缀硬预算、正负反事实收益平衡、16/64/128 三时间尺度
utility，以及 AUPRC、Brier、ECE 和写率间隔。

10 seeds 的正式任务使用 64 B 等状态、64 步 held-out 延迟、上下文转置路由和
两轮重复 query。MQR 准确率为 `1.0000`，相对 identity、GRU、fast-weight 的
配对差分别为 `0.43379 [0.42028,0.44729]`、
`0.49248 [0.48207,0.50289]`、`0.35176 [0.32264,0.38087]`；identity
replacement 显著降低性能，事件/干扰写率为 `1.000/0.000`，参数、状态、前向、
摊销更新和峰值解析资源门均通过。

然而直接固定环形缓冲区也为 `1.0000`，MQR-minus-fixed-ring 为精确的 0。
因此该实验把“当前 MQR 路由是否被行为使用”从否推进到是，却没有证明其
Cayley/unistochastic 部分优于普通离散环。下一门必须使用未知或上下文变化的
稀疏拓扑、匹配事件/干扰边缘分布并学习路由；固定 ring 不再能预编码答案后，才可
检验 MQR 独立性。完整分析见 `analysis/mqr_routed_memory_report.md`。

```bash
python3 test_routed_memory.py
python3 experiments/mqr_routed_memory_qualification.py \
  --seeds 10 --seed-start 71 --workers 4 \
  --output analysis/results/mqr_routed_memory_qualification_formal_10seed.json
python3 analysis/mqr_routed_memory_verify.py \
  analysis/results/mqr_routed_memory_qualification_formal_10seed.json
```
