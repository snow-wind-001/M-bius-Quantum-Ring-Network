(() => {
    "use strict";
    const byId = (id) => document.getElementById(id);
    const status = byId("toolbar-status");
    const proofs = [...document.querySelectorAll("details.proof, details.appendix")];
    byId("expand-proofs").addEventListener("click", () => {
        proofs.forEach((item) => { item.open = true; });
        status.textContent = "已展开全部推导与原报告。";
    });
    byId("collapse-proofs").addEventListener("click", () => {
        proofs.forEach((item) => { item.open = false; });
        status.textContent = "已收起补充推导；正文与主要公式仍可阅读。";
    });
    byId("print-report").addEventListener("click", () => window.print());
    let beforePrint = null;
    window.addEventListener("beforeprint", () => {
        if (beforePrint !== null) return;
        beforePrint = proofs.map((item) => item.open);
        proofs.forEach((item) => { item.open = true; });
    });
    window.addEventListener("afterprint", () => {
        if (beforePrint === null) return;
        proofs.forEach((item, i) => { item.open = beforePrint[i]; });
        beforePrint = null;
    });
    byId("download-data").addEventListener("click", () => {
        const blob = new Blob([byId("report-data").textContent], {type: "application/json;charset=utf-8"});
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "go13-report-data.json";
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
        status.textContent = "已导出本页结果数据与原始文件 SHA-256 索引。";
    });

    const toc = byId("toc");
    const narrow = window.matchMedia("(max-width: 880px)");
    const setToc = () => { toc.open = !narrow.matches; };
    setToc();
    narrow.addEventListener("change", setToc);
    const links = [...document.querySelectorAll(".toc a")];
    links.forEach((link) => link.addEventListener("click", () => {
        const target = byId(link.hash.slice(1));
        if (target) target.focus({preventScroll: true});
    }));
    if ("IntersectionObserver" in window) {
        const observer = new IntersectionObserver((entries) => {
            for (const entry of entries) {
                if (!entry.isIntersecting) continue;
                links.forEach((link) => {
                    if (link.hash === "#" + entry.target.id) link.setAttribute("aria-current", "location");
                    else link.removeAttribute("aria-current");
                });
            }
        }, {rootMargin: "0px 0px -65% 0px", threshold: 0});
        document.querySelectorAll("article > h2").forEach((heading) => observer.observe(heading));
    }
    // A numerical illustration of rho^k, not a measured model accuracy.
    const rhoInput = byId("decay-rho");
    const stepsInput = byId("decay-steps");
    const decayX = (k) => 48 + 600 * k / 128;
    const decayY = (value) => 210 - 160 * value;
    function updateDecay() {
        const rho = Number(rhoInput.value);
        const steps = Number(stepsInput.value);
        const retained = rho ** steps;
        let points = "";
        for (let k = 0; k <= 128; k++) points += decayX(k) + "," + decayY(rho ** k) + " ";
        byId("decay-curve").setAttribute("points", points.trim());
        byId("decay-point").setAttribute("cx", decayX(steps));
        byId("decay-point").setAttribute("cy", decayY(retained));
        byId("decay-cursor").setAttribute("x1", decayX(steps));
        byId("decay-cursor").setAttribute("x2", decayX(steps));
        byId("decay-step-label").textContent = steps;
        byId("decay-result").textContent =
            "经过 " + steps + " 步，状态差异／协向量长度剩余 " +
            (retained * 100).toFixed(2) + "%；归一化旋转乘积的长度保持为 100%。" +
            (rho === 0 ? "ρ = 0 时，旧状态在第一步就被清除。" : "这是几何保留比例，不是历史判断准确率。");
    }
    rhoInput.addEventListener("change", updateDecay);
    stepsInput.addEventListener("input", updateDecay);
    updateDecay();

    // Exact finite-step margins for m(delta)-b = delta_y - kappa*delta_x^2.
    const kappaInput = byId("curvature-kappa");
    const directionInput = byId("curvature-direction");
    const graphX = (x) => 65 + 610 * x / 1.05;
    const graphY = (y) => 265 - 195 * (y + 0.45) / 2.05;
    function updateCurvature() {
        const kappa = Number(kappaInput.value);
        const interior = directionInput.value === "interior";
        const dy = interior ? 0.12 : 0;
        const rows = [];
        let accepted = null;
        for (let step = 0; step <= 8; step++) {
            const alpha = 2 ** (-step);
            const margin = alpha * dy - kappa * (alpha * 0.8) ** 2;
            const feasible = margin >= -1e-5;
            if (accepted === null && feasible) accepted = {step, alpha, margin};
            rows.push({step, alpha, margin, feasible});
        }
        const xMax = kappa > 0 ? Math.min(1.05, Math.sqrt(1.6 / kappa)) : 1.05;
        let path = "";
        for (let i = 0; i <= 100; i++) {
            const x = xMax * i / 100;
            path += (i ? " L " : "M ") + graphX(x) + " " + graphY(kappa * x * x);
        }
        byId("curvature-curve").setAttribute("d", path);
        byId("curvature-region").setAttribute("d",
            path + " L " + graphX(xMax) + " " + graphY(1.6) +
            " L " + graphX(0) + " " + graphY(1.6) + " Z");
        const point = byId("curvature-direction-point");
        point.setAttribute("cy", graphY(dy));
        const ray = byId("curvature-ray");
        ray.setAttribute("y2", graphY(dy));
        const chosen = accepted === null ? {alpha: 0, step: null, margin: 0} : accepted;
        byId("curvature-accepted").setAttribute("cx", graphX(chosen.alpha * 0.8));
        byId("curvature-accepted").setAttribute("cy", graphY(chosen.alpha * dy));
        byId("curvature-kappa-label").textContent = kappa.toFixed(1);
        byId("curvature-result").textContent = accepted === null
            ? "8 次折半后仍不满足数值容差，提交零更新。最后候选的真实余量为 " +
              rows[8].margin.toExponential(3) + "，低于 −1 × 10⁻⁵。"
            : "第 " + accepted.step + " 次折半时通过检查：α = " + accepted.alpha +
              "，真实余量为 " + accepted.margin.toExponential(3) +
              "。允许误差为 1 × 10⁻⁵。";
        byId("curvature-attempt-rows").innerHTML = rows.map((row) =>
            "<tr" + (accepted && accepted.step === row.step ? ' class="highlight"' : "") + ">" +
            "<td>" + row.step + "</td><td>" + row.alpha + "</td><td>" +
            row.margin.toExponential(4) + "</td><td>" +
            (accepted && row.step > accepted.step ? "通过后不再尝试" : row.feasible ? "通过" : "拒绝") +
            "</td></tr>"
        ).join("");
    }
    kappaInput.addEventListener("input", updateCurvature);
    directionInput.addEventListener("change", updateCurvature);
    updateCurvature();
})();
