"""Audit completed Go14 runs while other independent runs are still executing."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.go14_verify import audit_learning, audit_games
from experiments import go14_readability as study


if __name__ == "__main__":
    study.check_lock()
    pending = {(stage, method, seed) for stage, methods in (
        ("learning", study.METHODS), ("games", study.GAME_METHODS))
        for method in methods for seed in study.SEEDS}
    running = {}
    target = ROOT / "analysis/results/go14_audit_execution.json"
    receipt = json.loads(target.read_text()) if target.exists() else []
    # Already-passed audits used the same or stricter numeric comparisons;
    # retain them only when both their input result and checkpoint still match.
    from experiments.go13_algorithm import sha
    for key in list(pending):
        stage, method, seed = key
        audit = ROOT / "analysis/results" / f"go14_{stage}_audit_{method}_seed{seed}.json"
        result = ROOT / "analysis/results" / f"go14_{stage}_{method}_seed{seed}.json"
        if audit.exists() and result.exists():
            saved = json.loads(audit.read_text())
            if saved.get("verified") and saved.get(stage + "_sha256") == sha(result):
                if stage == "games" or saved["checkpoint_sha256"] == sha(study.DIRECTORY / f"learned-{method}-{seed}.pt"):
                    pending.remove(key)
    with ProcessPoolExecutor(max_workers=2) as pool:
        while pending or running:
            for key in sorted(pending):
                if len(running) >= 2:
                    break
                stage, method, seed = key
                path = ROOT / "analysis/results" / f"go14_{stage}_{method}_seed{seed}.json"
                if path.exists():
                    function = audit_learning if stage == "learning" else audit_games
                    running[pool.submit(function, (method, seed))] = key
                    pending.remove(key)
            if not running:
                time.sleep(1)
                continue
            done, _ = wait(running, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                key = running.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    receipt.append({"key": key, "verified": False, "error": repr(error)})
                    target.write_text(json.dumps(receipt, indent=2) + "\n")
                    raise
                receipt.append({"key": key, "verified": True})
                target.write_text(json.dumps(receipt, indent=2) + "\n")
                print(json.dumps({"stage": key[0], "method": key[1], "seed": key[2],
                                  "verified": result["verified"]}), flush=True)
