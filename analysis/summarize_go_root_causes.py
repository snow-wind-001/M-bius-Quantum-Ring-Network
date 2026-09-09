"""Validate and summarize retrospective diagnostics without retraining models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def mean(values) -> float:
    return float(np.mean(list(values)))


def main() -> None:
    paths = [ROOT / "analysis/results/go_root_cause_diagnostics.json",
             ROOT / "analysis/results/go_spatial_alias_diagnostics.json"]
    probes, aliases = [json.loads(path.read_text()) for path in paths]
    for payload in (probes, aliases):
        assert payload["training_updates"] == 0 and payload["retrospective_diagnostic"]
        assert [run["seed"] for run in payload["runs"]] == [17, 29, 43, 71, 101]
        assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
                   for name, digest in payload["source_sha256"].items())
    summary = {}
    for method in probes["runs"][0]["methods"]:
        rows = [run["methods"][method] for run in probes["runs"]]
        gradients = [item for row in rows for item in row["gradient_windows"]]
        anchors = [item for row in rows for item in row["ogd_anchor_diagnostics"]]
        for row in rows:
            assert 1 <= row["centered_state_covariance_participation_rank"] <= 48
            for metrics in row["phase_probe_metrics"].values():
                assert abs(metrics["initial_joint_nll"] - metrics["joint_nll"]
                           - metrics["pass_component_nll_gain"] - metrics["conditional_point_component_nll_gain"]) < 1e-6
        for gradient in gradients:
            for key in ("raw_gradient_energy_by_block", "projected_gradient_energy_by_block"):
                assert abs(sum(gradient[key].values()) - 1) < 1e-6
            assert -1.00001 <= gradient["objective_joint_nll_gradient_cosine"] <= 1.00001
            assert 0 <= gradient["retained_norm"] <= 1.00001
        for anchor in anchors:
            assert 0 <= anchor["current_truncated_gradient_coverage_by_old_basis"] <= 1.00001
            assert 0 <= anchor["current_contextual_gradient_coverage_by_old_basis"] <= 1.00001
        item = {key: mean(row[key] for row in rows) for key in (
            "latent_tanh_derivative_mean", "latent_tanh_derivative_below_0_1_fraction",
            "injection_tanh_derivative_mean", "centered_state_covariance_participation_rank",
            "current_player_board_cosine_lag1", "canonical_black_white_board_cosine_lag1")}
        item["transition_normalized_drift"] = np.mean([row["transition_normalized_frobenius_drift_from_initial"] for row in rows], axis=0).tolist()
        item["raw_gradient_energy_fraction"] = {
            key: mean(gradient["raw_gradient_energy_by_block"].get(key, 0) for gradient in gradients)
            for key in ("heads", "modulation", "injection", "core_readout", "transition")}
        for key in ("objective_joint_nll_gradient_cosine", "projected_joint_nll_gradient_cosine",
                    "raw_direction_joint_nll_first_order", "projected_direction_joint_nll_first_order",
                    "norm_matched_raw_direction_joint_nll_first_order", "retained_norm"):
            item[key] = mean(gradient[key] for gradient in gradients)
        item["gradient_windows"] = len(gradients)
        item["phase_probe_means"] = {
            phase: {key: mean(row["phase_probe_metrics"][phase][key] for row in rows)
                    for key in rows[0]["phase_probe_metrics"][phase]} for phase in ("a", "b")}
        if anchors:
            item["anchors"] = {key: mean(anchor[key] for anchor in anchors) for key in anchors[0]}
            item["anchors"]["count"] = len(anchors)
        summary[method] = item
    alias_summary = {
        phase: {key: mean(run["phase_diagnostics"][phase][key] for run in aliases["runs"])
                for key in aliases["runs"][0]["phase_diagnostics"][phase]} for phase in ("a", "b")}
    count = sum(sum(item["positions"] for item in run["methods"]["global/orthogonal"]["phase_probe_metrics"].values())
                for run in probes["runs"])
    changed = sum(run["methods"]["global/orthogonal"]["teacher_label_changes_when_superko_history_erased"] for run in probes["runs"])
    for run in aliases["runs"]:
        assert run["empty_board_point_logit_range"] == 0
        assert run["empty_board_raw_model_action"] == 0
        assert run["empty_board_target_probability"] <= .040001
    result = {
        "valid": True, "diagnostic_only": True,
        "averaging": "state and phase summaries average five seed means; gradient and anchor summaries pool fixed probes",
        "probe_position_records": count, "teacher_labels_changed_by_superko_history_erasure": changed,
        "sampling_note": "records are reused across methods; board positions can recur and are not independent statistical samples",
        "teacher_label_change_fraction": changed / count,
        "methods": summary, "spatial_aliases": alias_summary,
        "inputs_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "source_files_match_workspace": True,
    }
    output = ROOT / "analysis/results/go_root_cause_summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Validated {count} position records, {len(summary)} methods, and five seeds; {output}")


if __name__ == "__main__":
    main()
