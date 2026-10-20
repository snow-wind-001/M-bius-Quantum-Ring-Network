"""Execute the separately locked task-order follow-up without replacing Go13."""
import argparse
import concurrent.futures
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.go13_algorithm import SEEDS,METHODS,sha
from experiments.go13_transfer import source_hashes


def main(workers):
    locked=json.loads((ROOT/'analysis/results/go13_transfer_source.json').read_text())
    assert locked['source']==source_hashes() and locked['plan_sha256']==sha(ROOT/'analysis/go13_transfer_plan.md')
    logs=ROOT/'checkpoints/go13_transfer/logs';logs.mkdir(parents=True,exist_ok=True)
    path=ROOT/'analysis/results/go13_transfer_execution.json'
    result={'source':source_hashes(),'started_utc':datetime.now(timezone.utc).isoformat(),'workers':workers,'jobs':[],'completed':False}
    def run(job):
        stage,seed,method=job
        cmd=[sys.executable,'experiments/go13_transfer.py','--stage',stage,'--seed',str(seed),'--method',method]
        log=logs/f'{stage}-{method}-{seed}.log';start=time.perf_counter()
        environment={**os.environ,'MKL_THREADING_LAYER':'GNU','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        with log.open('w') as stream:p=subprocess.run(cmd,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        return {'stage':stage,'seed':seed,'method':method,'command':cmd,'exit_code':p.returncode,'seconds':time.perf_counter()-start,'log':str(log.relative_to(ROOT))}
    for stage in ('prepare','learn','games'):
        jobs=[(stage,s,m) for s in SEEDS for m in (('orth_current',) if stage=='prepare' else METHODS)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for future in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
                row=future.result();result['jobs'].append(row);path.write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k!='command'}),flush=True)
                if row['exit_code']:raise RuntimeError(row['log'])
        assert source_hashes()==locked['source']
    result.update(completed=True,finished_utc=datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6)
    main(parser.parse_args().workers)
