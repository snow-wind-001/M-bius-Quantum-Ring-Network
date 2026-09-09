#!/usr/bin/env python3
"""Verify the local MiniCPM slow-LoRA consolidation smoke result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result",
        nargs="?",
        type=Path,
        default=Path("analysis/results/unified_temporal_mqr_minicpm_go_smoke.json"),
    )
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["experiment"] == "unified_temporal_mqr_minicpm_go"
    assert payload["mqr_effective"] is False
    lora = payload["lora"]
    assert lora["opportunities"] == payload["config"]["updates"]
    assert 0 < lora["updates"] <= lora["scheduled"] < lora["opportunities"]
    assert lora["adapter_drift"]["max_abs"] > 0.0
    assert lora["adapter_drift"]["changed_tensors"] > 0
    assert lora["frozen_parameter_probe_max_abs_drift"] == 0.0
    invariant = payload["invariants"]
    assert invariant["pending_tickets"] == 0
    assert invariant["prediction_before_update"] is True
    assert invariant["backbone_frozen"] is True
    assert invariant["max_unitary_error"] < 1e-4
    assert invariant["max_stochastic_error"] < 1e-4
    for phase in ("before", "after"):
        for value in payload["metrics"][phase].values():
            assert math.isfinite(float(value))
    print(
        json.dumps(
            {
                "scheduled": lora["scheduled"],
                "updates": lora["updates"],
                "adapter_max_abs_drift": lora["adapter_drift"]["max_abs"],
                "frozen_probe_max_abs_drift": lora[
                    "frozen_parameter_probe_max_abs_drift"
                ],
                "mqr_effective": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
