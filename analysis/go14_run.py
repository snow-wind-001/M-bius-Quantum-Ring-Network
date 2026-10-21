"""Execute the locked Go14 matrix; resume only completed, source-matched jobs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import go14_readability as study


def main(workers: int) -> None:
    lock = study.check_lock()
    if lock.get("phase") != "formal" or lock["seeds"] != list(study.SEEDS):
        raise RuntimeError("formal source manifest required")
    directory = study.DIRECTORY / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    target = ROOT / "analysis/results/go14_execution.json"
    receipt = json.loads(target.read_text()) if target.exists() else []

    def run(stage: str, seed: int, method: str) -> dict:
        command = [sys.executable, "experiments/go14_readability.py", "--stage", stage,
                   "--seed", str(seed)]
        if method:
            command += ["--method", method]
        log = directory / f"{stage}-{method or 'all'}-{seed}.log"
        started = time.perf_counter()
        with log.open("a") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        return {"stage": stage, "seed": seed, "method": method, "command": command,
                "exit_code": result.returncode, "seconds": time.perf_counter() - started,
                "log": str(log.relative_to(ROOT))}

    for stage, methods in (("prepare", ("",)), ("learn", tuple(study.METHODS)),
                           ("games", study.GAME_METHODS)):
        failed = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run, stage, seed, method)
                       for method in methods for seed in study.SEEDS]
            for future in as_completed(futures):
                row = future.result()
                receipt.append(row)
                target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
                print(json.dumps(row), flush=True)
                if row["exit_code"]:
                    failed.append(row)
        if failed:
            raise RuntimeError(f"{len(failed)} {stage} jobs failed; dependent stage was not started")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    main(parser.parse_args().workers)
