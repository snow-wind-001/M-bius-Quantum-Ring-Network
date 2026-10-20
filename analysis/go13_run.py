"""Run the locked Go13 matrix; each child is CPU single-threaded."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.go13_algorithm import SEEDS, METHODS, source_hashes, sha


def main(workers):
    source = source_hashes()
    locked = json.loads((ROOT / 'analysis/results/go13_source.json').read_text())
    if source != locked['source'] or sha(ROOT / 'analysis/go13_plan.md') != locked['plan_sha256']:
        raise RuntimeError('formal source or protocol changed')
    logs = ROOT / 'checkpoints/go13_v1/logs'; logs.mkdir(parents=True, exist_ok=True)
    receipt = ROOT / 'analysis/results/go13_execution.json'
    result = {'source': source, 'workers': workers, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'jobs': [], 'completed': False}
    def run(job):
        stage, seed, method = job
        command = [sys.executable, 'experiments/go13_algorithm.py', '--stage', stage, '--seed', str(seed), '--method', method]
        log = logs / f'{stage}-{method}-{seed}.log'; start = time.perf_counter()
        with log.open('w') as stream:
            process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        return {'stage': stage, 'seed': seed, 'method': method, 'command': command, 'exit_code': process.returncode,
                'seconds': time.perf_counter() - start, 'log': str(log.relative_to(ROOT))}
    for stage in ('prepare', 'learn', 'games'):
        jobs = [(stage, seed, method) for seed in SEEDS for method in (('orth_current',) if stage == 'prepare' else METHODS)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for future in concurrent.futures.as_completed([pool.submit(run, job) for job in jobs]):
                row = future.result(); result['jobs'].append(row)
                receipt.write_text(json.dumps(result, indent=2) + '\n')
                print(json.dumps({k:v for k,v in row.items() if k != 'command'}), flush=True)
                if row['exit_code']:
                    raise RuntimeError(f"failed job: {row['log']}")
        if source_hashes() != source:
            raise RuntimeError('source changed during formal execution')
    result.update(completed=True, finished_utc=datetime.now(timezone.utc).isoformat())
    receipt.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1: parser.error('workers must be positive')
    main(args.workers)
