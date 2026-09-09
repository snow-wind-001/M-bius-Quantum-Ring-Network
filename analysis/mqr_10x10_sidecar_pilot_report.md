# 10×10 MQR Sidecar Pilot：概率修正、等资源核对与负结果

> 权威结果：`results/unified_temporal_mqr_go_10x10_factorized_5seed_pilot.json`。
> 本实验为 5-seed 缩短 pilot，不是论文所需的 10–20 seed 正式结果；其用途是验证
> 大棋盘路径、发现指标污染并决定是否值得扩大训练预算。

## 1. 本轮修复

历史统一实验固定 `input_dim=214`。10×10 的精确棋盘特征为
\(3\times100+6=306\) 维，因此旧路径会不可逆压缩。当前 `lossless` 编码直接保留
own/opponent/previous-move 三个平面和六个因果标量。ko/superko 配对也已从仅
3×3 扩展到任意 \(N\ge3\)：焦点处两分支的当前棋盘、轮次、上一步和全部编码特征
逐元素相同，仅历史状态决定立即回提是否合法。

更关键的修复是 pass 概率。旧实现把二元 `pass_logit` 与条件 placement logits
直接拼接后 softmax；二者不是同一种 logit，结果使合法率被 pass 严重虚高。当前
采用严格分解

\[
\log P(i)=\log\sigma(-s_{pass})+\log\operatorname{softmax}(z)_{i},
\qquad
\log P(pass)=\log\sigma(s_{pass}).
\]

若加入学习合法性软先验，先修正 \(z_i\)，再对条件落点重新归一化。pass 初始先验
设为 \(1/(N^2+1)\)，使初始联合策略与均匀动作先验同尺度。修复前的 5-seed pilot
中 MQR 对局 pass 率约 64.6%，所以那一版 raw 合法率不再具有解释力。

## 2. Pilot 协议

- 棋盘：10×10，lossless 306 维编码；
- seeds：7、17、29、43、71；
- 每任务 3 个训练 episode，2 个 probe episode，每条记录 4 步；
- 每阶段 2 局无规则 mask 对局；双轨迹 horizon 2；
- 只运行离线 AWR/持续学习阶段，未启用 actor-critic；
- 四核心共享路由、四头、损失、样本次序、反馈次数与 OGD rank；
- bootstrap 仅 1000 次，故区间只用于 pilot 诊断。

大棋盘核心宽度只根据解析参数/MAC 公式预先调整，未查看任务指标。资源为：

| core | 参数 | 持久状态标量 | 估算前向 MAC |
|---|---:|---:|---:|
| MQR unistochastic | 13,032 | 9 | 8,757 |
| MQR identity | 13,440 | 12 | 8,592 |
| GRU | 13,500 | 9 | 8,550 |
| fast-weight | 13,344 | 48 | 8,496 |

参数比 1.0359、MAC 比 1.0307，均通过 5% 门。状态比为 5.333，严格 1.05
等状态门失败，因此总有效性门在查看性能前就不能通过。

## 3. 修正后的真实结果

下表为 after-task-B 的 5-seed 均值。`raw legal` 与 `useful legal` 在 pass=0 时相同，
因而不再存在“全 pass 看似合法”的解释。

| 方法 | task-B probe 教师一致 | 对局 raw/useful 合法率 | pass 率 | 对局回报 | 历史焦点准确率 | A 正遗忘 loss |
|---|---:|---:|---:|---:|---:|---:|
| MQR | 0.000 | 0.0153 | 0.000 | -17.125 | 0.500 | 0.00973 |
| identity | 0.000 | 0.0277 | 0.000 | -16.825 | 0.500 | 0.00656 |
| GRU | 0.000 | 0.0552 | 0.000 | -16.425 | 0.500 | 0.01238 |
| fast-weight | 0.000 | 0.0215 | 0.000 | -16.975 | 0.500 | 0.00390 |

MQR 的历史合法性概率分离只有 `0.000992`，虽然符号为正，但分类准确率仍精确
为机会水平 0.5。其回报相对 identity、GRU、fast-weight 的配对均值差分别为
`-0.300`、`-0.700`、`-0.150`；95% pilot bootstrap 区间分别为
`[-0.650,-0.050]`、`[-1.125,-0.300]`、`[-0.350,0.000]`。没有一个方向支持
MQR 优势。

有效性门现在直接使用上表的动态对局合法率，而不是稀疏 task-A probe。
MQR useful-legality 相对 identity、GRU、fast-weight 的配对均值差为
`-0.01245/-0.03994/-0.00620`，区间为 `[-0.02814,-0.00005]`、
`[-0.07089,-0.01543]`、`[-0.01240,0.00000]`。每个门指标的 JSON 来源
路径都由验证器重算，因此动态指标与最终决策已完全对齐。

probe 稀疏局面上的 MQR raw 合法率为 1.0，但动态对局只有 0.0153，说明固定
probe 过易且不能替代真实状态分布。task-B 历史 probe 上所有方法的教师
一致率均为 0；动态对局中 MQR/identity 为 0，GRU/fast-weight 也仅为约
`0.0046/0.0016`。当前训练预算仍未形成有意义的基础 placement 学习。
`mqr_effective=false` 是必要结论，不是统计功效
不足时的保守措辞。

## 4. 结论与下一停止门

本轮建立了三个有效工程结论：10×10 lossless/ko-history 路径可运行；四头联合策略
现在概率自洽；修正后的未掩码指标能暴露真实失败。它没有建立 MQR 的算法优势，
也没有证明小模型通过在线对局学会围棋规则。

扩大到正式实验前，应先在不增加核心差异的条件下达到以下最低学习门：

1. 所有方法在 held-out 动态局面上的有效合法落子率明显高于随机策略，而非只在
   稀疏固定 probe 上合法；
2. placement 教师一致率非零且 pass Brier 有校准改善；
3. ko/fresh 历史焦点在至少一个有状态核心上可靠超过 0.5；
4. 用状态预算上限或共享状态表示解决 9/12/9/48 的严格资源不匹配；
5. 达到上述门后再运行 10–20 seeds、更多真实劫形、固定强度 Sayuri 对手与短 RL。

若增加训练曝光后四核心都仍接近零合法率，下一步应先更换共享的空间编码器或
课程，而不是向 MQR 叠加更多环、RL 或慢 LoRA；否则无法归因任何改进。

## 5. 复现

```bash
python3 test_unified_mqr_agent.py
python3 experiments/unified_temporal_mqr_go.py \
  --board-size 10 --encoder-mode lossless \
  --seeds 7 17 29 43 71 \
  --train-episodes 3 --probe-episodes 2 --recorded-moves 4 \
  --twin-horizon 2 --eval-games 2 --bootstrap-resamples 1000 \
  --skip-actor-critic \
  --output analysis/results/unified_temporal_mqr_go_10x10_factorized_5seed_pilot.json
```
