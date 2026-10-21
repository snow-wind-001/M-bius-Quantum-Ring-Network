#!/usr/bin/env python3
"""Build the offline Go13 explanation from frozen evidence using Pandoc MathML."""

from __future__ import annotations

import base64
import hashlib
import html
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis/go13_report.html"
BASELINE = "0e78808751f563bdfdfd50ef1cf405b943c8da98"
METHODS = {
    "orth_free": "正交／无保护",
    "orth_current": "正交／当前约束",
    "similar_free": "相似变换／无保护",
    "similar_current": "相似变换／当前约束",
    "orth_stale": "正交／8 步刷新",
    "orth_norm": "正交／步长对照",
    "orth_frozen": "正交／冻结",
}
SECTIONS = [
    ("overview", "先看结论，再读证明"),
    ("protocol", "我们究竟测了什么"),
    ("results", "实验结果对比"),
    ("symbols", "符号与数据流"),
    ("orthogonality", "正交传播的保证"),
    ("covectors", "行为约束怎样回传"),
    ("projection", "投影与传播的协调"),
    ("similarity", "稳定非正交控制"),
    ("parameter-gradient", "从状态走到参数"),
    ("online-protection", "在线保护的优化问题"),
    ("finite-step", "为什么更新被拒绝"),
    ("write-bottleneck", "共享写入的秩限制"),
    ("observability", "历史信息能否被读出"),
    ("stability", "保护与学习的边界"),
    ("next", "证据判断与下一步"),
    ("evidence", "原报告与证据索引"),
]


def read_json(name: str) -> dict:
    return json.loads((ROOT / "analysis/results" / (name + ".json")).read_text())


def percentage(value: float, suffix: str = "%", signed: bool = False) -> str:
    number = (Decimal(str(value)) * 100).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)
    return ("+" if signed and number >= 0 else "") + str(number) + suffix


def markdown(source: str, prefix: str = "") -> str:
    """Render TeX to native MathML; refuse silent formula fallbacks."""
    command = [
        "pandoc", "-f",
        "markdown+tex_math_dollars+tex_math_single_backslash+fenced_divs+header_attributes",
        "-t", "html5", "--mathml",
    ]
    if prefix:
        command.append("--id-prefix=" + prefix)
    result = subprocess.run(command, input=source, text=True, capture_output=True, check=True)
    if result.stderr.strip():
        raise RuntimeError("Pandoc reported a conversion problem:\n" + result.stderr)
    output = re.sub(r"<p>(<summary>.*?</summary>)</p>", r"\1", result.stdout, flags=re.S)
    if "$$" in output or 'class="math display"' in output:
        raise ValueError("An equation was not converted into native MathML")
    return output


def number_equations(source: str) -> str:
    """Give display equations linkable labels without relying on TeX tag support."""
    numbers = []

    def replace(match: re.Match) -> str:
        formula = match.group(1)
        tag = re.search(r"\\tag\{(\d+)\}", formula)
        if tag is None:
            raise ValueError("Every display equation must have a unique number")
        number = int(tag.group(1))
        numbers.append(number)
        formula = re.sub(r"\\tag\{\d+\}", "", formula).strip()
        return (
            '\n\n::: {.equation #eq-' + str(number) + '}\n\n$$\n' + formula +
            '\n$$\n\n<a class="eq-number" href="#eq-' + str(number) +
            '" aria-label="公式 ' + str(number) + '">(' + str(number) + ')</a>\n\n:::\n\n'
        )

    result = re.sub(r"\$\$(.*?)\$\$", replace, source, flags=re.S)
    if numbers != list(range(1, 63)):
        raise ValueError("Expected the 62 documented equations in sequence")
    return result


def table(headers: list[str], rows: list[list[str]], caption: str = "",
          classes: str = "", highlighted: set[int] | None = None) -> str:
    highlighted = highlighted or set()
    result = '<table class="' + classes + '">'
    if caption:
        result += "<caption>" + caption + "</caption>"
    result += "<thead><tr>" + "".join("<th scope=\"col\">" + cell + "</th>" for cell in headers)
    result += "</tr></thead><tbody>"
    for i, row in enumerate(rows):
        result += '<tr class="highlight">' if i in highlighted else "<tr>"
        result += "".join("<td>" + cell + "</td>" for cell in row) + "</tr>"
    return result + "</tbody></table>"


def method_label(key: str) -> str:
    return '<span class="method-name">' + METHODS[key] + "<code>" + key + "</code></span>"


def decay_demo() -> str:
    return """
<section class="interactive" aria-labelledby="decay-title">
  <div class="interactive-header">
    <h4 id="decay-title">泄漏如何影响信用长度</h4>
    <span class="demo-label">可操作的公式示意 · 非实验成绩</span>
  </div>
  <p class="note">保持旋转，不等于取消泄漏。选择本轮实际使用的 ρ，观察同一初始差异的长度变化。</p>
  <div class="controls">
    <label for="decay-rho">泄漏系数 ρ
      <select id="decay-rho">
        <option value="0">0</option><option value="0.5">0.5</option>
        <option value="0.85">0.85</option><option value="0.93">0.93</option>
        <option value="0.97">0.97</option><option value="0.99" selected>0.99</option>
      </select>
    </label>
    <label for="decay-steps">步数 k
      <input id="decay-steps" type="range" min="1" max="128" value="64">
      <output id="decay-step-label" for="decay-steps">64</output>
    </label>
  </div>
  <svg class="demo-svg" viewBox="0 0 700 255" role="img" aria-labelledby="decay-svg-title">
    <title id="decay-svg-title">状态长度保留率：归一化旋转恒为一，带泄漏传播为 rho 的 k 次方。</title>
    <path d="M48 50H648 M48 130H648 M48 210H648" fill="none" stroke="#d7ddd5"/>
    <path d="M48 50V210H648" fill="none" stroke="#81958b"/>
    <path d="M48 50H648" fill="none" stroke="#a87b45" stroke-width="2" stroke-dasharray="5 5"/>
    <text x="2" y="54">100%</text><text x="10" y="134">50%</text><text x="17" y="214">0%</text>
    <text x="45" y="234">0</text><text x="340" y="234">64</text><text x="630" y="234">128 步</text>
    <line id="decay-cursor" x1="348" x2="348" y1="50" y2="210" stroke="#adb9ad" stroke-dasharray="3 4"/>
    <polyline id="decay-curve" points="" fill="none" stroke="#285e53" stroke-width="3"/>
    <circle id="decay-point" cx="348" cy="126" r="5" fill="#285e53"/>
  </svg>
  <div class="legend"><span>带泄漏：ρᵏ</span><span class="secondary">归一化旋转：1</span></div>
  <p id="decay-result" class="demo-result" aria-live="polite">ρ = 0.99 时，64 步后保留 52.56%。这是长度比例，不是任务准确率。</p>
  <noscript><p>启用 JavaScript 可拖动参数；以上正文与全部数学推导仍可离线阅读。</p></noscript>
</section>
"""


def curvature_demo() -> str:
    return """
<section class="interactive" aria-labelledby="curvature-title">
  <div class="interactive-header">
    <h4 id="curvature-title">线性边界与真实可行域</h4>
    <span class="demo-label">二维示意 · 不是实际训练轨迹</span>
  </div>
  <p class="note">绿色区域满足真实下界。切线投影只要求向上分量不为负，可能仍停留在真实曲线下面。</p>
  <div class="controls">
    <label for="curvature-kappa">曲率 κ
      <input id="curvature-kappa" type="range" min="0" max="4" step="0.1" value="2">
      <output id="curvature-kappa-label" for="curvature-kappa">2.0</output>
    </label>
    <label for="curvature-direction">候选方向
      <select id="curvature-direction">
        <option value="tangent">切线投影 (0.8, 0)</option>
        <option value="interior">带余量方向 (0.8, 0.12)</option>
      </select>
    </label>
  </div>
  <svg class="demo-svg" viewBox="0 0 710 305" role="img" aria-labelledby="curvature-svg-title">
    <title id="curvature-svg-title">一阶约束为 delta y 不小于零，真实约束为 delta y 不小于 kappa 乘 delta x 的平方。</title>
    <path id="curvature-region" d="" fill="#dfeadc"/>
    <path d="M65 70V265H675" fill="none" stroke="#a1afa3"/>
    <path d="M65 222.195H675" stroke="#81958b" stroke-dasharray="5 5" fill="none"/>
    <path id="curvature-curve" d="" fill="none" stroke="#285e53" stroke-width="3"/>
    <text x="74" y="88">真实可行域</text>
    <text x="48" y="227">0</text><text x="16" y="75">δy</text><text x="660" y="287">δx</text>
    <text x="73" y="211">一阶边界</text>
    <line id="curvature-ray" x1="65" y1="222.195" x2="529.762" y2="222.195" stroke="#a87b45" stroke-width="2"/>
    <circle id="curvature-direction-point" cx="529.762" cy="222.195" r="5" fill="#a87b45"/>
    <circle cx="529.762" cy="250.732" r="5" fill="#757e77"/>
    <text x="545" y="255">原候选</text>
    <circle id="curvature-accepted" cx="65" cy="222.195" r="6" fill="#285e53" stroke="#fff" stroke-width="2"/>
  </svg>
  <div class="legend"><span>真实边界／提交点</span><span class="secondary">被检查的方向</span><span class="dashed">一阶边界</span></div>
  <p id="curvature-result" class="demo-result" aria-live="polite">κ = 2 的切线方向在 8 次折半后仍不满足容差，提交零更新。</p>
  <details class="attempts">
    <summary>查看全部缩步计算</summary>
    <table><thead><tr><th>折半次数</th><th>步长 α</th><th>真实余量 m − b</th><th>检查结果</th></tr></thead>
      <tbody id="curvature-attempt-rows"></tbody>
    </table>
  </details>
</section>
"""


def build() -> None:
    primary = read_json("go13_summary")
    transfer = read_json("go13_transfer_summary")
    geometry = read_json("go13_geometry")
    readout = read_json("go13_readout_diagnostic")
    write = read_json("go13_write_diagnostic")
    drift = read_json("go13_policy_drift")
    validation = read_json("go13_validation")
    assert primary["verified"] and transfer["verified"] and geometry["verified"] and write["verified"]
    assert not transfer["primary_results_replaced"] and transfer["same_seed_cohort"]
    assert primary["complete_games"] + transfer["complete_games"] == 1820
    assert validation["root_tests_passed"] == 186

    replacements = {}
    facts = [
        ("1,820", "完整 10×10 对局<br>两轮每方法各 120 局评估"),
        ("5", "共同训练种子<br>补充沿用同一组种子"),
        ("36,191", "模型总参数<br>32,577 个可更新"),
        ("186", "既有根测试通过<br>本页整理未重跑训练"),
    ]
    replacements["FACTS"] = '<div class="facts">' + "".join(
        '<div class="fact"><strong>' + number + "</strong><span>" + label + "</span></div>"
        for number, label in facts
    ) + "</div>"
    chain = [
        ("01 写入", "历史差异进入共享表征"),
        ("02 传播", "正交通道保留几何"),
        ("03 读出", "差异变成正确判断"),
        ("04 约束", "限制已掌握行为漂移"),
        ("05 学习", "可行更新带来新能力"),
    ]
    replacements["LOGIC_CHAIN"] = '<ol class="logic-chain">' + "".join(
        "<li><b>" + label + "</b><small>" + detail + "</small></li>" for label, detail in chain
    ) + "</ol>"
    replacements["PROTOCOL"] = """
<div class="protocol">
  <div class="protocol-row">
    <div class="protocol-label">原矩阵<br>历史 → 局部</div>
    <div><div class="protocol-flow"><span>共同空间先验</span> → <span>A：2,048 条劫反馈</span> → <span>B：1,024 条局部反馈</span> → <span>完整棋局评估</span></div>
    <small>每种子 A 更新 128 次、B 更新 64 次；留出集各 256 条。先验 24 局、训练 24 epoch，另有 4 局完整探针。</small></div>
  </div>
  <div class="protocol-row">
    <div class="protocol-label">顺序补充<br>局部 → 新历史</div>
    <div><div class="protocol-flow"><span>继承学完 B 的参数与 Adam</span> → <span>16 条共同正确旧锚点</span> → <span>2,048 条全新劫反馈</span> → <span>完整棋局评估</span></div>
    <small>每种子 128 次更新尝试，保护下界为初始正确 margin 的一半；新历史留出 256 条。数据与原矩阵无重叠。</small></div>
  </div>
</div>
"""
    rows = []
    for key in METHODS:
        row = primary["table"][key]
        rows.append([
            method_label(key), percentage(row["a_before_accuracy"]), percentage(row["a_after_accuracy"]),
            percentage(row["b_before_accuracy"]) + " → " + percentage(row["b_after_accuracy"]),
            str(row["anchors"]), str(row["wins"]) + "/" + str(row["games"]),
        ])
    replacements["PRIMARY_TABLE"] = table(
        ["方法", "学完 A 的劫判断", "再学 B 后的劫判断", "B 局部规则判断", "历史锚点总数", "完整比赛胜局"],
        rows, "原矩阵 · 准确率为五种子平均；锚点与比赛为五种子合计。", "results-table", {1},
    )
    rows = []
    for key in METHODS:
        row = transfer["table"][key]
        denominator = row["attempted_updates"]
        accepted = str(row["accepted_updates"]) + "/" + str(denominator) if denominator else "0（冻结）"
        rows.append([
            method_label(key),
            percentage(row["old_before_accuracy"]) + " → " + percentage(row["old_after_accuracy"]),
            percentage(row["new_after_accuracy"]),
            str(row["actual_anchor_violations"]) + "/" + str(row["anchors"]),
            accepted, str(row["wins"]) + "/" + str(row["games"]),
        ])
    replacements["TRANSFER_TABLE"] = table(
        ["方法", "旧局部规则留出", "新劫历史留出", "最终锚点违规", "接受／尝试更新", "补充比赛胜局"],
        rows, "任务顺序补充 · 同一组五种子；每种子 16 个锚点、128 个反馈窗口。冻结组跳过参数更新。", "results-table", {1},
    )
    comparison = transfer["comparisons"]["orth_current_minus_orth_free"]
    cells = []
    for field, title in [
        ("old_retention_accuracy_change", "旧任务保持"),
        ("new_learning_accuracy_change", "新历史学习"),
    ]:
        item = comparison[field]
        cells.append(title + "差值 " + percentage(item["mean"], "", True) + " 个百分点 [" +
                     ", ".join(percentage(x, "", True) for x in item["interval95"]) + "]")
    replacements["COMPARISON_NOTE"] = '<p class="note">当前正交约束 − 正交无保护：' + "；".join(cells) + (
        "。95% 区间按五个训练种子穷举 3,125 次配对重采样，仅作小样本描述。"
        "“旧能力保留较好”与“新历史学会了”是两个判据，不能互相替代。</p>"
    )
    image_bytes = (ROOT / "analysis/figures/go13_summary.png").read_bytes()
    image_uri = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    replacements["RESULT_FIGURE"] = (
        '<figure><img width="2200" height="1200" src="' + image_uri +
        '" alt="Go13 原历史学习与顺序补充结果图；完整数值在相邻两张表中。">' +
        "<figcaption>本轮原图。阴影为五种子的实际范围；补充散点对应每个种子，叉号为均值。"
        "原矩阵与补充起点不同，比赛胜率分别解释。离线探针未混入在线成绩。</figcaption></figure>"
    )
    symbols = r"""
| 符号 | 维度／范围 | 含义 |
|:--|:--|:--|
| $x_t,\ c(x_t)$ | 306 维 | 当前观察；固定黑白坐标编码 |
| $B_\theta$ | $8\times306$ | 六环共享的输入压缩矩阵 |
| $u_t$ | 8 维 | 经过 tanh 的共享写入 |
| $U_j$ | $16\times8$ | 第 j 环的写入升维矩阵 |
| $h_{t,j},\ h_t$ | 16 维；96 维 | 单环状态；六环拼接状态 |
| $R_{t,j}$ | $16\times16$ | 当前输入条件化的实数正交旋转 |
| $\rho_j,\ \gamma_j$ | 固定标量 | 旧状态泄漏；$\gamma_j=\sqrt{1-\rho_j^2}$ |
| $\theta$ | 32,577 维 | 本轮实际参与在线更新的共享参数 |
| $J_t,\ G_t$ | $96\times96$；$96\times32577$ | 对旧状态与对参数的局部 Jacobian |
| $q_t$ | 96 维 | 终端行为对中间状态的协向量 |
| $a_i=\nabla_\theta m_i$ | 32,577 维 | 锚点 i 的当前参数行为法向量 |
| $m_i,\ b_i,\ c_i$ | 标量 | 正确 margin、保护下界、当前剩余余量 |
| $d,\ \delta$ | 32,577 维 | Adam 提出的实际步；经过约束处理的步 |
| $V,\ P_T,\ P_0$ | 按对应状态／参数空间定义 | 受保护方向的正交基；末端与早期投影 |
"""
    replacements["SYMBOL_TABLE"] = markdown(symbols).replace("<table>", '<table class="symbol-table">')
    replacements["DECAY_DEMO"] = decay_demo()
    replacements["CURVATURE_DEMO"] = curvature_demo()
    error = max(row["factorization_max_error"] for row in write["rows"])
    ranks = {row["conditional_rank_min"] for row in write["rows"]} | {
        row["conditional_rank_max"] for row in write["rows"]
    }
    assert ranks == {8}
    replacements["WRITE_FINDING"] = (
        '<p class="lead-note">五种子 × 两种几何，每配置 128 对留出历史，条件秩全部为 8；'
        "因子分解最大误差为 " + format(error, ".2e") +
        "。这批输入中没有出现 |tanh| &gt; 0.95 的饱和，所以“输入已饱和”不符合当前诊断。</p>"
    )
    rows = []
    for key, label in [("orth_after_a", "正交"), ("similar_after_a", "相似变换")]:
        item = readout["summary"][key]
        rows.append([label, percentage(item["model_accuracy"]),
                     percentage(item["offline_probe_accuracy"]), percentage(item["reset_probe_accuracy"])])
    replacements["PROBE_TABLE"] = table(
        ["学完 A 的模型", "原在线模型", "固定特征离线探针", "重置历史探针"], rows,
        "五种子均值；线性 ridge 正则 λ = 1，未使用留出集调参。"
    )
    replacements["CLAIM_TABLE"] = table(["研究命题", "当前判断", "证据边界"], [
        ["正交能改善状态传播几何", '<span class="status proof-status">支持</span>',
         "固定轨迹的状态通道；有泄漏时仍衰减，不等于任意网络或参数空间等距。"],
        ["当前参数约束能限制已保存行为漂移", '<span class="status measured-status">支持</span>',
         "真实任务冲突下补充实验零锚点违规；稳定相似变换也能保护。"],
        ["多环会自动增加可读历史容量", '<span class="status hypothesis-status">未建立</span>',
         "本轮两次观察共享 8 维写入，末层探针仍约 54%。"],
        ["少量反馈可获得可靠新历史判断", '<span class="status hypothesis-status">未实现</span>',
         "当前保护组 50.47%，大部分更新回退到零。"],
        ["有限锚点保证整体不遗忘", '<span class="status hypothesis-status">不支持</span>',
         "锚点达标时，未保存旧任务仍退化。"],
        ["几何优势已转化为联合任务优势", '<span class="status hypothesis-status">未建立</span>',
         "需同时有新任务收益与旧任务保持；本轮没有满足。"],
        ["已具备完整 RSI 或自主改进", '<span class="status hypothesis-status">未实现</span>',
         "固定任务、真实外部反馈、给定边界；尚无自主提出并验证改进的闭环。"],
    ])
    sources = [
        ("analysis/go13_report.md", "原始完整结论、资源成本、局限与复现命令"),
        ("analysis/go13_math.md", "原始数学契约及推导"),
        ("analysis/go13_plan.md", "原矩阵的固定实验协议"),
        ("analysis/go13_transfer_plan.md", "失败触发的任务顺序补充协议"),
        ("analysis/results/go13_summary.json", "原矩阵：五种子 × 七方法"),
        ("analysis/results/go13_transfer_summary.json", "任务顺序补充：五种子 × 七方法"),
        ("analysis/results/go13_geometry.json", "40 组固定历史几何与伴随核对"),
        ("analysis/results/go13_write_diagnostic.json", "两次观察共享写入因子分解与条件秩"),
        ("analysis/results/go13_readout_diagnostic.json", "冻结特征的离线读出诊断"),
        ("analysis/results/go13_policy_drift.json", "相同棋谱上的策略行为漂移"),
        ("analysis/results/go13_validation.json", "原研究：186 项根测试与完整棋谱核验"),
        ("mqr/go_research.py", "当前行为法向量、实际参数步投影与真实行为检查"),
        ("mqr/constraint_transport.py", "状态约束与参数梯度的传播实现"),
    ]
    replacements["SOURCE_TABLE"] = table(
        ["材料", "本页使用的证据"],
        [['<a href="' + ("../" if not path.startswith("analysis/") else "") +
          path.removeprefix("analysis/") + '">' + html.escape(Path(path).name) + "</a>", detail]
         for path, detail in sources],
        "原始研究基线：Git " + BASELINE[:8] + "。本页补充解释，不覆盖原结果。",
        "source-table",
    )
    original = markdown((ROOT / "analysis/go13_report.md").read_text(), "original-")
    original = re.sub(r"<h([1-5])(\b[^>]*)>(.*?)</h\1>",
                      lambda m: "<h" + str(int(m[1]) + 1) + m[2] + ">" + m[3] +
                      "</h" + str(int(m[1]) + 1) + ">", original, flags=re.S)
    original = original.replace('src="figures/go13_summary.png"', 'src="' + image_uri + '"')
    replacements["ORIGINAL_REPORT"] = (
        '<details class="appendix" id="original-report"><summary>展开完整原报告（保留所有表格与复现命令）</summary>\n' +
        original + "\n</details>"
    )
    source_paths = [path for path, _ in sources] + [
        "analysis/go13_explained.md", "analysis/figures/go13_summary.png",
        "analysis/go13_html_assets/report.css", "analysis/go13_html_assets/report.js",
        "scripts/build_go13_report.py",
    ]
    hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths}
    manifest_rows = [[html.escape(path), "<code>" + digest + "</code>"] for path, digest in hashes.items()]
    replacements["REPRODUCIBILITY"] = (
        '<details class="appendix"><summary>文件指纹与本页生成方式</summary>'
        "<p>使用项目内命令 <code>python3 scripts/build_go13_report.py</code> 重建。"
        "需要 Python 3.9+ 与 Pandoc 的 MathML 输出；阅读生成后的 HTML 不需要这些依赖。"
        "本页说明稿、结果表和指纹一起构成可复查材料。</p>" +
        table(["输入文件", "SHA-256"], manifest_rows, classes="source-table") +
        "<p>186 项测试和 1,820 局核验来自原研究记录；本次只新增解释文档与展示验证，"
        "没有重新训练模型。可从页首导出嵌入结果及完整指纹。</p></details>"
    )
    source = (ROOT / "analysis/go13_explained.md").read_text()
    body = markdown(number_equations(source))
    for token, replacement in replacements.items():
        placeholder = "{{" + token + "}}"
        body = body.replace("<p>" + placeholder + "</p>", replacement)
        body = body.replace(placeholder, replacement)
    body = re.sub(
        r"(<table\b.*?</table>)",
        r'<div class="table-scroll" tabindex="0" role="region" aria-label="可横向滚动的数据表">\1</div>',
        body, flags=re.S,
    )
    body = re.sub(r'<h2 id="([^"]+)">', r'<h2 id="\1" tabindex="-1">', body)
    if re.search(r"\{\{[A-Z_]+\}\}", body):
        raise ValueError("An HTML content token was not resolved")
    if body.count('class="eq-number"') != 62:
        raise ValueError("Missing numbered equations")
    nav = "".join(
        '<li><a href="#' + anchor + '"><span>' + str(i).zfill(2) + "</span>" + title + "</a></li>"
        for i, (anchor, title) in enumerate(SECTIONS, 1)
    )
    bundle = {
        "scope": "Go13 explanation artifact; frozen original results, no new training claims",
        "research_baseline_commit": BASELINE,
        "seeds": [911, 919, 929, 937, 941],
        "primary": {key: primary[key] for key in ("verified", "table", "comparisons", "complete_games")},
        "transfer": {key: transfer[key] for key in (
            "verified", "table", "comparisons", "complete_games", "primary_results_replaced", "same_seed_cohort")},
        "geometry": geometry,
        "write_diagnostic": write,
        "readout_summary": readout["summary"],
        "policy_drift_summary": drift["summary"],
        "source_sha256": hashes,
    }
    data = json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    css = (ROOT / "analysis/go13_html_assets/report.css").read_text()
    script = (ROOT / "analysis/go13_html_assets/report.js").read_text()
    page = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Go13 正交环在线学习研究解释：实验结果、62 个编号公式、逐步推导与互动示例。">
<title>正交环如何学会历史 · Go13 研究与数学推导</title>
<style>""" + css + """</style>
</head>
<body>
<a class="skip" href="#main">跳到报告正文</a>
<div class="shell">
  <aside class="sidebar" aria-label="报告导航">
    <a class="brand" href="#main">Möbius Quantum Ring</a>
    <div class="edition">RESEARCH NOTE / GO13</div>
    <nav aria-label="章节目录"><details class="toc" id="toc" open>
      <summary>阅读目录 · 16 节</summary><ol>""" + nav + """</ol>
    </details></nav>
    <p class="sidebar-foot">正交环 · 行为约束 · 在线学习<br>围棋作为可核验环境<br>研究基线 0e78808</p>
  </aside>
  <main id="main">
    <header class="hero">
      <div class="eyebrow">研究解释 / GO13 <span class="offline">离线可读</span></div>
      <h1>正交环如何学会历史</h1>
      <p class="subtitle">从状态传播到在线行为保护，逐步解释当前算法的数学结构、实验发现与尚未闭合的学习链条。</p>
      <div class="hero-meta"><span>更新于 2026-10-21</span><span>62 个编号公式</span><span>五种子 · 七方法 · 两轮研究</span></div>
      <div class="toolbar" role="group" aria-label="阅读工具">
        <button id="expand-proofs" type="button">展开全部推导</button>
        <button id="collapse-proofs" type="button">收起补充推导</button>
        <button id="print-report" type="button">打印 / 保存 PDF</button>
        <button id="download-data" type="button">导出结果数据</button>
      </div>
      <div id="toolbar-status" class="toolbar-status" aria-live="polite">建议先读结果，再沿数学主线阅读；点击公式编号可定位。</div>
      <p class="print-hint">宽表格和长公式可左右滑动查看。</p>
    </header>
    <article aria-label="研究报告正文">""" + body + """</article>
    <footer class="foot">MQR / Go13 · 本页是已有研究的解释版。数学保证、实验发现与待验证方向分别标明。<br>
    原始研究基线 0e78808 · 单文件 HTML，内含公式、图表与结果数据。</footer>
  </main>
</div>
<script type="application/json" id="report-data">""" + data + """</script>
<script>""" + script + """</script>
</body>
</html>
"""
    OUTPUT.write_text(page)
    print(json.dumps({
        "output": str(OUTPUT), "bytes": OUTPUT.stat().st_size,
        "numbered_equations": 62, "native_mathml_nodes": page.count("<math "),
        "source_files_hashed": len(hashes), "external_runtime_dependencies": 0,
    }, ensure_ascii=False))


if __name__ == "__main__":
    build()
