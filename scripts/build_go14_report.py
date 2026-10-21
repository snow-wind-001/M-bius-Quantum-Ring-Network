"""Build the Go14 Chinese report and standalone offline MathML HTML from audited data."""
from __future__ import annotations

import base64
import hashlib
import html
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"
MAIN = ("spatial_free", "spatial_current", "spatial_reserve", "spatial_adaptive",
        "spatial_rollback", "spatial_frozen", "spatial_aligned", "spatial_rule24")
LABELS = {"shared": "共享写入", "grouped": "分组三块写入", "spatial_pool": "位置写入／平均读出",
          "spatial": "位置写入／位置查询", "previous": "上一可见观察寄存器", "similar_spatial": "位置查询／相似变换",
          "kernel_pool": "卷积写入／平均读出", "kernel": "卷积写入／位置查询", "similar_kernel": "卷积查询／相似变换"}


def pct(x):
    return str((Decimal(str(x)) * 100).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)) + "%"


def table(headers, rows):
    return "\n| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "\n".join(
        "| " + " | ".join(str(x) for x in row) + " |" for row in rows) + "\n"


def charts(summary):
    directory = OUT / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#faf9f6", "axes.facecolor": "#faf9f6"})
    layouts = summary["layouts"]
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    names = list(layouts)
    y = np.arange(len(names))
    ax.barh(y, [layouts[n]["accuracy"]["mean"] * 100 for n in names],
            color=["#285e53" if n in ("spatial", "kernel") else "#b7c5b7" for n in names])
    for i, name in enumerate(names):
        values = np.asarray(layouts[name]["accuracy"]["per_seed"]) * 100
        ax.scatter(values, i + np.linspace(-.13, .13, 5), s=15, color="#26352e", zorder=3)
    ax.axvline(50, color="#806135", ls="--", lw=1)
    ax.set(yticks=y, yticklabels=names, xlim=(0, 103), xlabel="Held-out history accuracy (%) · dots: five seeds")
    ax.invert_yaxis()
    fig.savefig(directory / "go14_readability.svg")
    fig.savefig(directory / "go14_readability.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.4, 5), layout="constrained")
    colors = plt.get_cmap("tab10")
    for i, method in enumerate(MAIN):
        row = summary["table"][method]
        ax.scatter(row["a_after"] * 100, row["b_after"] * 100, s=60, color=colors(i), label=method)
    limit = max(80, (summary["table"]["spatial_free"]["a_before"] - .05) * 100)
    ax.axvline(limit, color="#285e53", ls="--", lw=1)
    ax.axhline(90, color="#285e53", ls="--", lw=1)
    ax.set(xlabel="Old history accuracy after learning B (%)", ylabel="New rule accuracy (%)",
           xlim=(45, 102), ylim=(45, 103))
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title("Stability and continued learning · five-seed means", fontsize=12)
    fig.savefig(directory / "go14_joint.svg")
    fig.savefig(directory / "go14_joint.png", dpi=160)
    plt.close(fig)


def report(summary):
    layouts, rows = summary["layouts"], summary["table"]
    action_probe = json.loads((OUT / "results/go14_action_probe.json").read_text())
    assert action_probe["source"] == summary["source"] and action_probe["post_hoc_read_only"]
    quality = json.loads((OUT / "results/go14_policy_quality.json").read_text())
    assert quality["source"] == summary["source"] and quality["post_hoc_rescoring_only"]
    passed = [name for name, row in rows.items() if row["joint_gate"]]
    chunks = ["# Go14：历史读出改善，整体在线链条尚未闭合\n",
        "2026-10-21 · 五个新种子 · 10×10 围棋 · 独立核验后的研究记录\n",
        "## 本轮结论\n",
        f"**位置化写入与位置查询解决了本轮的历史读出失败。** 相同参数预算下，共享写入的留出准确率为 {pct(layouts['shared']['accuracy']['mean'])}，位置查询为 {pct(layouts['spatial']['accuracy']['mean'])}；同样扩大状态但平均读出的对照为 {pct(layouts['spatial_pool']['accuracy']['mean'])}。上一观察寄存器对照为 {pct(layouts['previous']['accuracy']['mean'])}。这表明任务确实可以学会，也表明增加环状态本身不足以解决问题。位置模型到 1,024 条反馈时仍仅有 51.80%，所以少量反馈高效学习尚未实现。\n",
        ("联合门槛通过的方法为：" + "、".join(f"`{name}`" for name in passed) + "。这只支持当前固定任务、预算与五种子均值，不构成任意分布不遗忘的保证。\n") if passed else
        "**尚无方法通过预定联合门槛。** 本轮建立了可学的历史能力，并检验了真实的旧能力保护；仍未完成稳定的‘新判断提升＋旧判断保持’链条。\n",
        f"同时，**在线约束已有可测量的保持收益**：自适应方案的新规则准确率为 {pct(rows['spatial_adaptive']['b_after'])}，无保护为 {pct(rows['spatial_free']['b_after'])}；旧历史保持分别为 {pct(rows['spatial_adaptive']['a_after'])} 与 {pct(rows['spatial_free']['a_after'])}。自适应方案接受 {rows['spatial_adaptive']['accepted']}/320 次更新，实际平均步长 {rows['spatial_adaptive']['mean_step_norm']:.5f}，无保护为 {rows['spatial_free']['mean_step_norm']:.5f}。这是有新学习时的部分保持收益，尚未达到旧准确率下降不超过五个百分点的目标。\n",
        f"正交位置模型与稳定相似变换模型的历史准确率分别为 {pct(layouts['spatial']['accuracy']['mean'])} 和 {pct(layouts['similar_spatial']['accuracy']['mean'])}。当前证据支持可寻址读出的价值，未建立欧氏正交相对该稳定对照的独有任务优势。配对历史只改变 5/640 对原始贪心落子，说明行动转化仍弱。完整对局及未保存行为另列，不用历史题准确率代替棋力。\n",
        f"完整对局中，策略对齐方案为 {summary['games']['spatial_aligned']['wins']}/120 胜，冻结参照为 {summary['games']['spatial_frozen']['wins']}/120 胜，自适应方案为 {summary['games']['spatial_adaptive']['wins']}/120 胜；本轮没有建立超出冻结参照的棋力收益。\n",
        "## 实验边界与公平比较\n",
        "正式种子固定为 1009、1013、1019、1021、1031，开发种子 4903 排除。每种子先用 24 局完整教师棋谱训练共同空间先验 24 epoch，另有四局独立探针棋谱。随后 A 为 2,048 条劫历史反馈，B 为 1,024 条自杀／可落子反馈；每 16 条反馈更新一次，分别为 128 次与 64 次机会。A→B 保留 Adam 状态。留出 A、B 各 256 条；每条历史两次观察，从规则合法的设置局面构造并平衡两种标签，属于结构化同生成分布测试，并非自然对局中的历史频率。先预测再接收反馈，训练流单次访问。\n",
        "九种结构均有 36,191 个总参数、32,577 个声明可更新参数，学习张量形状与初始化值逐一相同。不同前向路径激活的参数作用不完全相同：寄存器对照使用既有修正头的另一半输入，位置查询消融移除相应信息通路。因此这是相同参数预算比较，不是所有有效自由度、状态容量或计算成本都相等。位置状态从 96 增至 9,600 个 float32 坐标，单流内存从 384 增至 38,400 字节；寄存器对照为 2,840 字节。\n",
        "保护训练过且正确的历史锚点：四种可读环模型共同选出 12 对，按查询象限和执色分层；主比较使用前八对／16 条，`rule24` 使用全部 24 条。下界为各自初始正确 margin 的一半。策略锚点来自已见训练棋谱，未保存策略探针来自另外四局，均使用两次观察。\n",
        "源码与协议在正式训练前固定 SHA-256，开发结果全部单列。v1 因有限步入口与提交检查的 float32 边界不一致而中止；v2 统一数值判据，保持 1e-5 容差并从头重跑。除此实现勘误外，模型、参数、预算、锚点规则及种子未调整。统计以五个训练种子为单位；配对 bootstrap 枚举 3,125 个重采样，只作小样本描述，不能把 1,280 道留出题当作 1,280 次独立训练。\n",
        "## 一、写入与查询：历史是否真正学会\n",
        table(["结构", "历史准确率", "成对全对率", "重置历史", "关闭查询修正", "状态字节", "学习门槛"], [
            [f"{LABELS[name]} `{name}`", pct(row["accuracy"]["mean"]), pct(row["pair_accuracy"]["mean"]),
             pct(row["interventions"]["reset"]), pct(row["interventions"]["no_query"]), row["resources"]["state_bytes_float32"],
             "通过" if row["learned_gate"] else "未通过"] for name, row in layouts.items()]),
        "![历史准确率：柱为五种子均值，点为各个种子；虚线为 50%。](figures/go14_readability.svg)\n",
        "学习门槛预先固定为平均准确率至少 80%、成对全对率至少 60%。每对样本末次编码与查询相同、标签相反，因此仅看当前观察的确定性模型不能超过 50%；单纯重置历史的确得到 50%。交换历史前缀的成绩是原准确率的互补，不当作第二份独立证据。\n",
        table(["结构", "32 条反馈", "128 条", "512 条", "1,024 条", "2,048 条"], [
            [name] + [pct(row["curves"][i]["accuracy"]) for i in range(1, 6)] for name, row in layouts.items()]),
        "这里的判断是指定位置合法性的二分类，并非整盘最佳落子选择。位置模型在 512 条时为 50%、1,024 条时为 51.80%，到 2,048 条才达到 97.11%；上一观察对照在 1,024 条已达 93.28%。这些曲线表明当前环模型尚未具备所期望的少量反馈效率。A 的最终成绩是在共同棋盘先验之后、额外 2,048 条结构化反馈上获得；不能写成模型从零仅凭几局棋就掌握围棋。\n",
        "### 只读补充：历史判断是否转化为落子选择\n",
        "以下诊断在主实验锁定后提出，只读取 A 结束时的现有检查点，不改变训练、门槛或模型选择。用原始策略分数直接取贪心动作，不提供引擎合法掩码。配对比较的两条历史有相同当前编码；重置比较则可能引入与目标历史无关的状态分布变化。\n",
        table(["布局", "配对历史改变动作／640", "重置后改变动作／1280", "合法时选择查询点／640", "非法时选择查询点／640", "查询点概率差／百分点"], [
            [name, row["paired_raw_greedy_changes"], row["raw_greedy_changes_after_reset"],
             row["query_selected_when_legal"], row["query_selected_when_illegal"],
             f"{100 * row['mean_legal_minus_illegal_query_probability']:.3f}"] for name, row in action_probe["summary"].items()]),
        "位置模型只有 5/640 对因为真实历史差异改变原始贪心动作；查询点在合法分支只被选中两次。概率差方向改善，并不等于学会最佳落子；本任务只提供合法性标签，也没有证明合法回提在每道题中都是最佳手。重置造成 388/1280 个动作变化不能替代这一更严格的配对证据。‘历史判断已可学’与‘判断已经可靠指导围棋决策’应继续分开。\n",
        "## 二、学会之后：能否继续更新而不丢失旧判断\n",
        "以下每行是五种子平均；下降为旧任务 A 后到 B 后的准确率差，负数表示改善。主模型旧历史均值为 " + pct(rows["spatial_free"]["a_before"]) + "。联合门槛同时要求：旧准确率至少 80%、下降不超过五个百分点；新准确率至少 90%、比 B 前提高至少十个百分点。\n",
        table(["方法", "旧 A 前→后", "旧下降／百分点", "新 B 前→后", "联合门槛"], [
            [name, pct(row["a_before"]) + " → " + pct(row["a_after"]), f"{100 * (row['a_before'] - row['a_after']):.2f}",
             pct(row["b_before"]) + " → " + pct(row["b_after"]), "通过" if row["joint_gate"] else "未通过"]
            for name, row in rows.items()]),
        "![旧历史保持与新规则学习：右侧与上方虚线表示主模型联合门槛中的准确率条件。](figures/go14_joint.svg)\n",
        "`current` 是原真实 margin 检查；`reserve` 保留一半线性余量并对缩短的候选重新投影；`adaptive` 再加曲率估计与步长半径；`rollback` 仅在最终拒绝时恢复 Adam 状态。`aligned` 加策略行为，`rule24` 增加规则锚点。`free` 和 `frozen` 分别显示无保护遗忘与冻结上限。正交与相似变换的同布局控制同时保留。\n",
        "## 三、有效更新预算与成本\n",
        "每个非冻结方法累计 5,120 条 B 反馈、320 次尝试。接受只是提交了非零步，不保证有实质学习；表中实际步长是参数空间欧氏范数。耗时为六个训练进程并行时各种子记录的墙钟时间之和，不是专用硬件基准或整个研究总用时。\n",
        table(["方法", "接受／尝试", "激活约束", "缩步次数", "Adam 回退", "候选均范数", "实际均范数", "B 耗时／秒"], [
            [name, f"{row['accepted']}/{row['attempts']}", row["active_updates"], row["backtracks"], row["optimizer_rollbacks"],
             f"{row['mean_raw_norm']:.5f}", f"{row['mean_step_norm']:.5f}", f"{row['seconds']:.1f}"] for name, row in rows.items()]),
        "完整 JSON 另存每窗预测、真实标签、投影次数、投影收敛、最小非零步长与回退记录。规则锚点和法向量的存储远大于单条环状态；以下是实际保存的约束张量字节，不包含 Python 对象、临时自动微分图或进程峰值内存。\n",
        table(["主方法", "每种子规则／策略锚点", "约束张量字节／种子", "最终锚点违规数"], [
            [name, ", ".join(f"{a}/{b}" for a, b in zip(rows[name]["rule_anchors_per_seed"], rows[name]["policy_anchors_per_seed"])),
             ", ".join(map(str, rows[name]["constraint_tensor_bytes"])), rows[name]["anchor_violations"]] for name in MAIN]),
        "free/frozen 表中的规则锚点数用于事后对照，其学习器没有加载保护约束；因此约束存储为零。未保护的违规不是保护合同失败。\n",
        "## 四、受保护对象是否与整体行为对齐\n",
        "`aligned` 的策略锚点数依次为 " + "、".join(map(str, rows["spatial_aligned"]["policy_anchors_per_seed"])) +
        "，没有用不正确或 margin 不足的行为补齐。只有八条齐全的种子，`aligned` 与 `rule24` 的约束方向数才相同。以下 120 个探针来自未保存局面，衡量 A→B 的行为变化；教师一致率不是最优棋判断准确率。\n",
        table(["方法", "留出贪心动作改变／120", "策略 KL（前‖后）", "教师一致率前→后", "原始停着概率前→后"], [
            [name, row["policy_greedy_changes"], f"{row['policy_kl']:.4f}",
             pct(row["policy_teacher_before"]) + " → " + pct(row["policy_teacher_after"]),
             pct(row["raw_pass_before"]) + " → " + pct(row["raw_pass_after"])] for name, row in rows.items()]),
        "策略保护从 A 结束后开始，不能据此宣称 A 学习期间的原始先验策略也已得到保持。保存行为达标与未保存行为泛化是两种证据。保护的是选定上下文中的输出 margin，不是一个‘永久不遗忘’标签；被冻结的策略头仍会因共享特征变化而改变选择。\n",
        "动作变化本身还可能纠正旧错误。下面对同一批已保存预测作事后重评分，区分原有教师一致行为的保持与新增加的一致行为；不增加训练或挑选局面。教师一致仍只是代理指标，不能等同于最优落子。\n",
        table(["方法", "原有教师一致动作", "其中保持", "其中丢失", "新增加的一致动作"], [
            [name, quality["summary"][name]["teacher_correct_before"], quality["summary"][name]["retained_teacher_correct"],
             quality["summary"][name]["lost_teacher_correct"], quality["summary"][name]["new_teacher_correct"]] for name in MAIN]),
        "自适应规则保护只保留原有 23 个教师一致动作中的七个，加入策略锚点后保留十四个。这是保护对象对齐的局部迹象；与此同时新规则准确率由 99.30% 降到 86.25%，没有形成联合成功。五种子共 23 个原有一致动作的样本量很小，且不能把所有其他动作漂移都计作遗忘。\n",
        "## 五、完整 10×10 对局检查\n",
        "六个预定方法各 120 局，共 720 局冻结评估；加新共同先验与探针的 140 局，本轮正式记录总计 860 局。双方必须连续两次停着终局，没有达到步数上限后直接判胜。配对换色、4／24／60 手开局、两类启发式对手；16 次搜索模拟、深度八、值头贡献为零。其他学习控制没有运行完整比赛，不对它们报告胜率。\n",
        table(["方法", "胜局／120", "胜率", "平均目差", "原始贪心合法率", "对局耗时／秒"], [
            [name, f"{row['wins']}/{row['games']}", pct(row["win_rate"]), f"{row['mean_margin']:.2f}",
             pct(row["raw_legal_rate"]), f"{row['seconds']:.1f}"] for name, row in summary["games"].items()]),
        "搜索使用引擎合法动作约束及真实终局规则；其最终落子合法不能反过来证明网络原始判断可靠。这里的比赛检验模型与搜索的集成表现，没有在对局中更新参数。结构化劫任务的提升不能直接外推成自然长历史、一般超级劫能力或完整在线棋力学习。\n",
        "从任务定义可以解释两者为何未必同步：A、B 只监督指定位置的合法性，搜索已经通过引擎排除了非法动作；合法动作之间的价值排序没有在这轮得到真实新收益监督。策略头冻结也不能阻止共享特征漂移。这是训练目标与行动效果之间尚未补齐的环节。\n",
        "完整对局的环状态贯穿所有落子，A/B 训练却只有两次观察，因此评估还包含训练未覆盖的长期状态分布。低胜率同时混合了目标与动作排序不对齐、上下文分布变化和共同先验能力有限，不能单独归因为正交传播失效。后续应先让真实对局的训练和评估上下文一致，再逐步扩长；本轮未做这个额外对照。\n",
        "## 六、每个种子与配对差异\n",
        table(["布局／种子顺序", "1009", "1013", "1019", "1021", "1031"], [
            [name] + [pct(x) for x in row["accuracy"]["per_seed"]] for name, row in layouts.items()]),
        table(["主方法", "各种子旧 A 后", "各种子新 B 后"], [
            [name, ", ".join(pct(p["a_after"]) for p in rows[name]["per_seed"]),
             ", ".join(pct(p["b_after"]) for p in rows[name]["per_seed"])] for name in MAIN]),
        table(["左组减右组", "旧 A 差／百分点 [描述性 95% 区间]", "新 B 差／百分点 [描述性 95% 区间]"], [
            [name] + [f"{row[k]['mean'] * 100:.2f} [{row[k]['interval95'][0] * 100:.2f}, {row[k]['interval95'][1] * 100:.2f}]" for k in ("a", "b")]
            for name, row in summary["comparisons"].items()]),
        (OUT / "go14_math.md").read_text(),
        "## 下一轮应验证什么\n",
        "优先保留本轮已验证的位置查询，让新实验从真正学会的历史能力出发。然后在固定的 32／128／512 条预算上检验写入尺度、读出归一化和初始化；正交传播继续保留，不能把这些未实施的尝试写成已有少样本优势。与此同时保留更省状态内存的上一观察对照：它在 A 上更准，却在无保护继续学习后明显遗忘，说明保存信息与保持读出是两个问题。本轮没有给这个寄存器对照施加同等行为保护，因此尚不能排除‘简单外部记忆＋相同行为约束’更有竞争力。\n",
        "下一轮先加入相同参数预算的交错 A/B 或联合训练参照，区分模型能否同时表示两类规则与顺序更新路径的问题；反馈量、重用旧样本次数与成本分别报告。再在训练样本池中选择覆盖不同位置、任务行为和梯度方向的约束，保留独立留出集。不能把本轮退化全部归因于锚点数量，也不能凭增加锚点就宣布分布保持已解决。\n",
        "有限步方面应同时优化保持、学习与延迟。内部余量和自适应步长减少了冻结，但没有自动改善留出保持；优化器回退没有受到真正拒绝事件的充分考验。利用批独立性降低法向量计算成本可以保留当前数学合同，需另做等价性与实测耗时检查。\n",
        "判断到行动的下一项任务，应让两个同样可行的选择具有可验证的不同收益，并检查历史是否改变正确选择；规则学习则同时报告不依赖引擎掩码的原始判断和非法尝试。围棋真实终局、策略排序反馈与旧行为保持要接入同一个在线闭环，避免只训练辅助分类器。\n",
        "在上述链条达到预定标准后，再依次扩大真实历史长度并控制反馈次数与更新频率，检查状态版本一致性。完整 RSI 还需要提出候选改进、分配测试预算、使用独立验证并决定保留或回退；本轮固定任务与固定边界的更新系统尚未具备这一能力。以上是后续可检验方案，均未写成本轮结果。\n",
        "## 证据、复现与开发记录\n",
        "独立核验重建了五种子全部原始历史、训练／留出隔离和真实规则标签，检查 860 局完整棋谱的合法步、两次停着、目数及胜负；复算 110 个后续学习配置的第一窗更新和检查点最终输出。另从检查点复现 " + str(summary["full_matches_reproduced"]) + " 局完整学生搜索轨迹，并在其余种子抽查第一步，总计 " + str(summary["decisions_reproduced"]) + " 次搜索决策。检查点 SHA 保存在审计 JSON 中；检查点不提交 Git。\n",
        "22 条验证命令全部通过：199 项根目录测试、数学恒等式检查、离线 Digits 回归、快速示例与语法编译。策略概率复算使用 float32 容差（绝对 1e-7、相对 1e-6），动作必须完全一致；曲率估计含 float32 差分除以步长平方，复算采用绝对 1e-5、相对 1e-4 容差；接受／拒绝与计数必须完全一致。初版逐位比较和过严曲率诊断容差触发的核验失败记录保留在执行日志中，不影响或改写正式实验。\n",
        "开发种子 4903 的结果没有混入正式平均。弱先验为两局／两 epoch；正式先验设计为 24 局／24 epoch。强先验开发中，共享写入约 50.39%、位置查询约 96.09%、上一观察对照 100%；位置模型配少量高 margin 锚点后旧任务约 72%–73%，卷积共享约 55%–57%。这些失败促成正式协议中的分层共同锚点及完整对照，但不能据开发与正式结果的差异单独推断分层因果效果，二者种子与选择规则不同。\n",
        "[固定实验协议](go14_plan.md) · [汇总与每种子数值](results/go14_summary.json) · [源码锁定](results/go14_source.json) · [数值勘误](go14_numerical_amendment.patch) · [v1 中止证据](results/go14_v1_interrupted/go14_execution.json) · [验证命令与输出](results/go14_checks.json) · [运行记录](results/go14_execution.json) · [独立核验记录](results/go14_audit_execution.json) · [核验容差说明](results/go14_numerical_audit_notes.json) · [只读动作诊断](results/go14_action_probe.json) · [旧正确行为重评分](results/go14_policy_quality.json) · [Go13 原报告](go13_report.html)\n",
        "```bash\npython3 test_go_readable.py\npython3 analysis/go14_run.py --workers 6\npython3 analysis/go14_verify.py --stage seeds\npython3 analysis/go14_verify.py --stage learning\npython3 analysis/go14_verify.py --stage games\npython3 analysis/go14_verify.py --stage summary\npython3 analysis/go14_action_probe.py\npython3 analysis/go14_policy_drift.py\npython3 scripts/build_go14_report.py\n```\n",
        "已有同源完整检查点会跳过训练；源码或协议指纹不一致会拒绝混用。无检查点时从固定种子重新生成先验、任务与完整对局。在线实现是独立研究 API；既有默认模型与 Go13 冻结证据保持不变。\n"]
    return "\n".join(chunks)


def build():
    summary = json.loads((OUT / "results/go14_summary.json").read_text())
    assert summary["verified"] and summary["complete_games"] == 860
    checks = json.loads((OUT / "results/go14_checks.json").read_text())
    assert len(checks) == 22 and all(row["exit_code"] == 0 for row in checks)
    charts(summary)
    source = report(summary)
    (OUT / "go14_report.md").write_text(source)
    rendered = subprocess.run(["pandoc", "-f", "markdown+tex_math_dollars", "-t", "html5", "--mathml"],
                              input=source, text=True, capture_output=True, check=True)
    if rendered.stderr:
        raise RuntimeError(rendered.stderr)
    body = rendered.stdout
    for name in ("go14_readability", "go14_joint"):
        encoded = base64.b64encode((OUT / f"figures/{name}.svg").read_bytes()).decode()
        body = body.replace(f'src="figures/{name}.svg"', f'src="data:image/svg+xml;base64,{encoded}"')
    body = re.sub(r"(<table>.*?</table>)", r'<div class="table-scroll">\1</div>', body, flags=re.S)
    headers = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body)
    nav = "".join(f'<a href="#{html.escape(key)}">{label}</a>' for key, label in headers)
    css = """
    :root{color-scheme:light;--bg:#faf9f6;--ink:#24342f;--green:#285e53;--line:#d7ddd5}
    *{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}
    body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.95 'Noto Sans CJK SC','Microsoft YaHei',sans-serif}
    .shell{max-width:1370px;display:grid;grid-template-columns:220px minmax(0,930px);gap:56px;margin:auto;padding:30px 40px 90px}
    aside{position:sticky;top:20px;align-self:start;max-height:95vh;overflow:auto;font-size:12px}aside strong{font-size:18px}nav{margin:18px 0}summary{cursor:pointer;margin-top:18px;color:#285e53}nav a{display:block;padding:7px 0;text-decoration:none}
    main{min-width:0}h1,h2{font-family:'Noto Serif CJK SC',serif;font-weight:600;line-height:1.65}h1{font-size:34px;margin:20px 0 30px}h2{font-size:25px;border-top:1px solid var(--line);padding-top:32px;margin-top:65px}h3{font-size:19px;margin-top:32px}
    a{color:var(--green);text-underline-offset:3px}p{margin:20px 0}code{font-size:.85em;overflow-wrap:anywhere}pre{padding:20px;background:#eef1eb;overflow:auto}pre code{overflow-wrap:normal}
    .table-scroll{overflow:auto;border-block:1px solid var(--line);margin:26px 0}table{border-collapse:collapse;min-width:680px;width:100%;font-size:12px;line-height:1.7;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:12px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{background:#ecf0e8}tr:last-child td{border:0}
    figure{margin:28px 0}img{max-width:100%;height:auto}figcaption{font-size:12px;color:#59685f}math[display=block]{display:block;max-width:100%;overflow:auto;padding:20px 12px;margin:24px 0;background:#f0f3ed;font-size:1.08em}
    button,select{font:inherit;color:var(--ink);background:var(--bg);border:1px solid #8c9c90;padding:6px 10px}button{cursor:pointer}.demo{border-block:2px solid var(--green);padding:20px 0;margin-top:35px}.demo p{margin:12px 0}.demo-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:6px 24px;font-size:14px}.muted{font-size:12px;color:#5c6a60}.skip{position:absolute;top:-60px}.skip:focus{top:4px;left:4px;background:white;padding:4px 12px}:focus-visible{outline:3px solid #619d89;outline-offset:3px}
    @media(max-width:900px){.shell{display:block;padding:20px}aside{position:static;max-height:none}nav{display:flex;flex-wrap:wrap;gap:5px 18px;margin:10px 0 24px}nav a{font-size:11px}h1{font-size:28px}h2{font-size:22px}.demo-grid{display:block}}
    @media print{body{background:white;font-size:10pt}.shell{display:block;padding:0}aside,.demo,.skip{display:none}h1{font-size:23pt}h2{font-size:17pt;break-after:avoid}h3{break-after:avoid}.table-scroll{overflow:visible}table{min-width:0;font-size:7pt}th,td{padding:5px 4px}math[display=block]{overflow:visible;background:none;font-size:10pt}a{color:inherit}pre{white-space:pre-wrap}img{max-height:240px}tr{break-inside:avoid}}
    @media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
    """
    options = "".join(f'<option value="{name}"{ " selected" if name == "spatial_adaptive" else ""}>{name}</option>' for name in summary["table"])
    demo = '<section class="demo" aria-labelledby="demo-title"><h3 id="demo-title">检查一个方法是否同时学会与保持</h3><label for="method">选择实测方法 </label><select id="method">' + options + '</select><div id="joint-data" class="demo-grid" aria-live="polite"></div><p class="muted">只读取本轮实测均值。反馈预算相同不代表有效更新相同；是否通过使用正式实验前固定的门槛。</p></section>'
    next_section = '<h2 id="' + headers[1][0] + '">'
    body = body.replace(next_section, demo + next_section, 1)
    script = """
    const data=JSON.parse(document.getElementById('evidence').textContent);
    function update(){const r=data.table[document.getElementById('method').value];const p=x=>(100*x).toFixed(2)+'%';
    const rows=[['旧历史',p(r.a_before)+' → '+p(r.a_after)],['新规则',p(r.b_before)+' → '+p(r.b_after)],['旧能力下降',(100*(r.a_before-r.a_after)).toFixed(2)+' 个百分点'],['接受 / 尝试',r.accepted+' / '+r.attempts],['实际步长均值',r.mean_step_norm.toFixed(5)],['联合门槛',r.joint_gate?'通过':'未通过']];
    document.getElementById('joint-data').replaceChildren(...rows.map(([label,value])=>{const p=document.createElement('p');const b=document.createElement('strong');b.textContent=label+'：';p.append(b,document.createTextNode(value));return p}));}
    const wide=matchMedia('(min-width:901px)');const toc=document.getElementById('toc');toc.open=wide.matches;wide.addEventListener('change',event=>{toc.open=event.matches});
    document.getElementById('method').addEventListener('change',update);update();document.getElementById('print').addEventListener('click',()=>window.print());
    """
    document = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Go14 · 可读历史与在线行为保护</title><style>' + css + '</style></head><body><a class="skip" href="#content">跳到正文</a><div class="shell"><aside><strong>Möbius / Go14</strong><p class="muted">离线研究报告 · 原生 MathML</p><button id="print" type="button">打印 / 保存 PDF</button><details id="toc" open><summary>报告目录</summary><nav aria-label="报告目录">' + nav + '</nav></details></aside><main id="content">' + body + '</main></div><script type="application/json" id="evidence">' + json.dumps(summary, ensure_ascii=False).replace("</", "<\\/") + '</script><script>' + script + '</script></body></html>'
    target = OUT / "go14_report.html"
    target.write_text(document)
    assert document.count('<math') >= 100 and '$$' not in document
    print(json.dumps({"path": str(target), "bytes": target.stat().st_size, "math_nodes": document.count('<math'),
                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    build()
