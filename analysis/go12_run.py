"""Execute the preregistered Go12 matrix without model selection or resampling."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.go12_efficiency import SEEDS, METHODS, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    directory = ROOT / "checkpoints/go12_v1"
    logs = directory / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    record = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "source": source_hashes(), "workers": args.workers, "jobs": []}
    output = ROOT / "analysis/results/go12_execution.json"

    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(output)

    def execute(seed, method, prepare=False):
        command = [sys.executable, "experiments/go12_efficiency.py", "--seed", str(seed),
                   "--method", method, "--directory", str(directory)]
        if prepare:
            command += ["--prepare-only"]
        log = logs / f"{'prepare' if prepare else method}-{seed}.log"
        started = time.perf_counter()
        with log.open("w") as stream:
            process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        return {"seed": seed, "method": method, "prepare": prepare, "command": command,
                "exit_code": process.returncode, "seconds": time.perf_counter() - started,
                "log": str(log.relative_to(ROOT))}

    save()
    for preparing in (True, False):
        jobs = [(seed, "combined") for seed in SEEDS] if preparing else [
            (seed, method) for seed in SEEDS for method in METHODS]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(execute, seed, method, preparing) for seed, method in jobs]
            for future in as_completed(futures):
                result = future.result()
                record["jobs"].append(result)
                save()
                print(json.dumps(result), flush=True)
        if any(job["exit_code"] for job in record["jobs"]):
            raise RuntimeError("a job failed; inspect logs without changing formal parameters")
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    record["completed"] = True
    save()


if __name__ == "__main__":
    main()
