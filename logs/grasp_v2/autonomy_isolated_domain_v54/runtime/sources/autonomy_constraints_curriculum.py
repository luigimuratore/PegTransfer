"""Evaluate learned full motors while reducing explicit grasp assistance.

Firm/gentle grasp presets and an optional free-release insertion guide.
Same frozen weights at every strength. No teacher, automatic stage progress
or silent retraining. Stop at a failed gate; native promotion requires zero
support on both seeds and an independently completed native 3/3 pilot.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from autonomy_full_recovery import native_full_gate
from autonomy_constraints_gate import assisted_gate

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--pilot_report',type=Path,required=True)
    p.add_argument('--native_pilot_report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait_minutes',type=float,default=45.)
    p.add_argument('--support_profile',choices=('gentle','firm'),default='firm')
    p.add_argument('--placement_support',action='store_true')
    args=p.parse_args();digest=sha(args.policy)
    if args.wait_minutes<=0:p.error('Wait time must be positive')
    manifest=json.loads((args.policy.parent/'manifest.json').read_text())
    if digest!=manifest['policy_sha256']:raise ValueError('Weights changed')
    for name,expected in manifest['source_hashes'].items():
        if sha(args.policy.parent/'sources'/name)!=expected:raise ValueError('Frozen source changed')
    if manifest['physics_manifest_sha256']!=sha(ROOT/'milestones/checkpoint3-full-assisted.json'):raise ValueError('Physics changed')
    native=json.loads(args.native_pilot_report.read_text())
    if native.get('policy_sha256')!=digest or not native_full_gate(native,3,1.):raise ValueError('Same-weight completed native pilot required')
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,args.output/'curriculum_source.py')
    shutil.copy2(HERE/'autonomy_full_recovery.py',args.output/'native_gate_source.py')
    shutil.copy2(HERE/'autonomy_constraints_gate.py',args.output/'assisted_gate_source.py')
    shutil.copy2(HERE/'autonomy_support.py',args.output/'support_source.py')
    status=dict(complete=False,native_full_policy_verified=False,policy=str(args.policy),policy_sha256=digest,
        algorithm='Frozen learned BC policy; explicit grasp and optional placement constraints',
        support_profile=args.support_profile,placement_support=args.placement_support,strengths=[1.,.5,.25,.1,0.],
        conditions=dict(sources=['L5'],targets=['R2'],jitter_mm=.25,yaw_deg=1.),results=[],stage='waiting_for_assisted_pilot')
    def save():
        status['updated_utc']=datetime.now(timezone.utc).isoformat();path=args.output/'status.json';temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(path)
    try:
        save();deadline=time.monotonic()+args.wait_minutes*60
        pilot_close=args.pilot_report.parent/'environment_close.json'
        while not (args.pilot_report.exists() and pilot_close.exists()):
            error=args.pilot_report.parent/'error.json'
            if error.exists():raise RuntimeError(error.read_text())
            if time.monotonic()>deadline:raise RuntimeError('Assisted pilot wait expired')
            time.sleep(10)
        pilot=json.loads(args.pilot_report.read_text())
        closed=json.loads(pilot_close.read_text())
        if (closed.get('environment_closed') is not True or closed.get('evaluation_complete') is not True
            or closed.get('evaluation_sha256')!=sha(args.pilot_report)):
            raise ValueError('Pilot must complete and close its environment cleanly')
        if pilot.get('policy_sha256')!=digest or not assisted_gate(pilot,1.,3,1.,args.support_profile,args.placement_support):
            status.update(complete=True,stage='assisted_pilot_failed');save();return 1
        status['pilot_report']=str(args.pilot_report);status['pilot_report_sha256']=sha(args.pilot_report)
        zero=[]
        for strength in status['strengths']:
            for seed in (123,9876):
                label=f'support_{str(strength).replace(".","p")}_seed{seed}';output=args.output/label
                status.update(stage=label,current_strength=strength);save()
                command=[sys.executable,'-u',str(HERE/'autonomy_replay.py'),'evaluate','--policy',str(args.policy),
                    '--skill','full','--grasp_support',str(strength),'--num_envs','16','--episodes','16','--episode_steps','10000',
                    '--sources','L5','--targets','R2','--jitter_mm','.25','--yaw_deg','1','--seed',str(seed),
                    '--headless','--diagnostic_events','--output',str(output)]
                command.extend(['--support_profile',args.support_profile])
                if strength and args.placement_support:command.append('--placement_support')
                with (args.output/(label+'.log')).open('w') as log:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);status['child_pid']=child.pid;save()
                    try:code=child.wait()
                    except BaseException:child.send_signal(signal.SIGINT);child.wait();raise
                status.pop('child_pid',None)
                report_path=output/'evaluation.json';report=json.loads(report_path.read_text())
                if code not in (0,1) or sha(args.policy)!=digest or report.get('policy_sha256')!=digest or report.get('seed')!=seed:raise ValueError('Incomplete or incompatible physical evaluation')
                passed=assisted_gate(report,strength,profile=args.support_profile,placement=args.placement_support)
                closed=json.loads((output/'environment_close.json').read_text())
                if passed and (code!=0 or closed.get('environment_closed') is not True
                    or closed.get('evaluation_complete') is not True or closed.get('evaluation_sha256')!=sha(report_path)):
                    raise ValueError('A passed evaluation must exit successfully with verified cleanup')
                row=dict(strength=strength,seed=seed,success=report['success'],episodes=report['episodes'],passed=passed,
                    report=str(report_path),report_sha256=sha(report_path));status['results'].append(row);save()
                if not passed:
                    status.update(complete=True,stage='gate_failed',failed_strength=strength,
                        reason='Retain current/reference checkpoints; collect targeted recoveries before reducing assistance');save();return 1
                if strength==0:zero.append(row)
        folder=args.output/'native_certification';folder.mkdir()
        shutil.copy2(__file__,folder/'certification_source.py')
        shutil.copy2(HERE/'autonomy_full_recovery.py',folder/'certification_gate_source.py')
        certified=dict(complete=True,native_full_policy_verified=True,stage='certified',policy=str(args.policy),policy_sha256=digest,
            pilot_report=str(args.native_pilot_report),pilot_report_sha256=sha(args.native_pilot_report),results=zero,
            conditions=status['conditions'],certification_source_hashes={name:sha(folder/name) for name in ('certification_source.py','certification_gate_source.py')})
        (folder/'status.json').write_text(json.dumps(certified,indent=2)+'\n')
        status.update(complete=True,stage='zero_support_verified',native_full_policy_verified=True,native_certification=str(folder));save();return 0
    except BaseException as error:
        status.update(complete=False,error=f'{type(error).__name__}: {error}');save();raise

if __name__=='__main__':raise SystemExit(main())
