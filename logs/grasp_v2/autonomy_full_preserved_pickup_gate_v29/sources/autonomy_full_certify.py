"""Certify a frozen native full candidate after its completed 3-case pilot."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import signal
import shutil
import subprocess
import sys
from autonomy_full_recovery import native_full_gate

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--pilot_report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();digest=sha(args.policy)
    manifest=json.loads((args.policy.parent/'manifest.json').read_text())
    if manifest['policy_sha256']!=digest:raise ValueError('Model changed')
    for name,expected in manifest['source_hashes'].items():
        if sha(args.policy.parent/'sources'/name)!=expected:raise ValueError('Frozen source changed')
    if manifest['physics_manifest_sha256']!=sha(ROOT/'milestones/checkpoint3-full-assisted.json'):raise ValueError('Physics changed')
    pilot=json.loads(args.pilot_report.read_text())
    if pilot.get('policy_sha256')!=digest or not native_full_gate(pilot,3,1.):raise ValueError('Completed native full 3/3 pilot required')
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,args.output/'certification_source.py')
    shutil.copy2(HERE/'autonomy_full_recovery.py',args.output/'certification_gate_source.py')
    status=dict(complete=False,native_full_policy_verified=False,policy=str(args.policy),policy_sha256=digest,
        pilot_report=str(args.pilot_report),pilot_report_sha256=sha(args.pilot_report),
        conditions=dict(sources=['L5'],targets=['R2'],jitter_mm=.25,yaw_deg=1.),
        requirement='At least 15/16 physically complete cases on each seed, same weights',results=[])
    status['certification_source_hashes']={name:sha(args.output/name) for name in ('certification_source.py','certification_gate_source.py')}
    def save():
        status['updated_utc']=datetime.now(timezone.utc).isoformat()
        path=args.output/'status.json';temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(path)
    try:
        for seed in (123,9876):
            output=args.output/f'seed{seed}';status['stage']=f'evaluate_seed{seed}';save()
            command=[sys.executable,'-u',str(HERE/'autonomy_replay.py'),'evaluate','--skill','full','--policy',str(args.policy),
                '--output',str(output),'--num_envs','16','--episodes','16','--episode_steps','10000',
                '--sources','L5','--targets','R2','--jitter_mm','.25','--yaw_deg','1','--seed',str(seed),'--headless']
            with (args.output/f'seed{seed}.log').open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                status['child_pid']=child.pid;save()
                try:code=child.wait()
                except BaseException:
                    child.send_signal(signal.SIGINT);child.wait();raise
            status.pop('child_pid',None)
            if code not in (0,1):raise RuntimeError(f'Evaluation exited {code}')
            report=json.loads((output/'evaluation.json').read_text())
            if sha(args.policy)!=digest or report.get('policy_sha256')!=digest:raise ValueError('Weights differ between evaluations')
            passed=native_full_gate(report,16,.9)
            status['results'].append(dict(seed=seed,success=report['success'],episodes=report['episodes'],passed=passed,
                report=str(output/'evaluation.json'),report_sha256=sha(output/'evaluation.json')))
            if not passed:
                status.update(complete=True,stage='not_promoted',reason='Native full evaluation failed; original checkpoints retained');save();return 1
        status.update(complete=True,stage='certified',native_full_policy_verified=True);save();return 0
    except BaseException as exc:
        status.update(complete=False,error=f'{type(exc).__name__}: {exc}');save();raise

if __name__=='__main__':raise SystemExit(main())
