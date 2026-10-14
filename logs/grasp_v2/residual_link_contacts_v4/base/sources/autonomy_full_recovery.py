"""Train from completed native full recoveries; certify only two-seed inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

REPO=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def native_full_gate(report,episodes=16,threshold=.9):
    rows=report.get('cases',[])
    if report.get('complete') is not True or report.get('skill')!='full':return False
    if report.get('teacher_at_inference') is not False or report.get('assisted_grasp') is not False:return False
    if len(rows)!=episodes or report.get('episodes')!=episodes:return False
    successes=[r for r in rows if r.get('success') and r.get('full_success') and r.get('handover')]
    if len(successes)!=report.get('success') or len(successes)!=report.get('full_success'):return False
    if len(successes)/episodes<threshold:return False
    for row in successes:
        physics=row.get('terminal_physics',{})
        if row.get('failure') or row.get('timeout') or row.get('peak_force_n',float('inf'))>1.:return False
        if row.get('hole_error_mm',float('inf'))>1.5:return False
        if not .008<=physics.get('bottom_m',float('inf'))<=.012:return False
        force=physics.get('forces_n',[])
        if len(force)!=2 or any(len(f)!=2 or any(abs(v)>.01 for v in f) for f in force):return False
    return True

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--datasets',type=Path,nargs='+',required=True)
    p.add_argument('--wait_minutes',type=float,default=45.)
    args=p.parse_args()
    if args.wait_minutes<=0:p.error('Positive waiting limit required')
    args.output.mkdir(parents=True,exist_ok=False)
    status=dict(complete=False,native_full_policy_verified=False,algorithm='Hierarchical Behavior Cloning with neural pressure feedback',
        stage='waiting_for_completed_native_recoveries',conditions=dict(source='L5',target='R2',jitter_mm=.25,yaw_deg=1.),
        certification='Pilot 3/3; then >=90% of 16 cases on each of two seeds with identical weights',steps=[])
    def save():
        status['updated_utc']=datetime.now(timezone.utc).isoformat()
        temporary=args.output/'status.json.tmp';temporary.write_text(json.dumps(status,indent=2)+'\n')
        temporary.replace(args.output/'status.json')
    def run(module,argv,stage):
        status['stage']=stage;save()
        command=[sys.executable,'-u',str(HERE/module),*map(str,argv)]
        print('[FULL RECOVERY] '+' '.join(command),flush=True)
        with (args.output/(stage+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            status['child_pid']=child.pid;save()
            try:code=child.wait()
            except BaseException:
                child.send_signal(__import__('signal').SIGINT);child.wait();raise
        status.pop('child_pid',None);status['steps'].append(dict(stage=stage,returncode=code));save()
        if code not in (0,1):raise RuntimeError(f'{stage} exited {code}; inspect its log')
        return code
    try:
        save();deadline=time.monotonic()+args.wait_minutes*60
        while True:
            missing=[path for path in args.datasets if not path.exists()]
            if not missing:break
            for path in missing:
                error=path.parent/'error.json'
                if error.exists():raise RuntimeError(f'Collection failed: {error.read_text()}')
                report=path.parent/'evaluation.json'
                if report.exists() and json.loads(report.read_text()).get('complete'):
                    raise RuntimeError(f'No successful native dataset from {path.parent}')
            if time.monotonic()>deadline:raise RuntimeError('Timed out waiting for complete recovery datasets')
            status['waiting_for']=[str(path) for path in missing];save();time.sleep(10)
        status.pop('waiting_for',None)
        datasets=[];provenance=[];folder=args.output/'datasets';folder.mkdir()
        baseline=hashlib.sha256((REPO/'milestones/checkpoint3-full-assisted.json').read_bytes()).hexdigest()
        for i,path in enumerate(args.datasets):
            manifest=json.loads((path.parent/'manifest.json').read_text())
            if manifest.get('physics_manifest_sha256')!=baseline:raise ValueError('Mixed physics manifests')
            report=json.loads((path.parent/'evaluation.json').read_text())
            if not report.get('complete') or report.get('assisted_grasp') is not False:raise ValueError('Incomplete or assisted collection')
            with np.load(path,allow_pickle=False) as archive:
                meta=json.loads(str(archive['metadata']))
                if meta.get('schema')!='peg-autonomy-bc-1' or meta.get('skill')!='full' or meta.get('assisted_grasp') is not False:
                    raise ValueError('Only native full demonstrations are accepted')
                if not meta.get('collection_complete') or len(meta.get('episodes',[]))<2:raise ValueError('Need completed successes with whole held-out episodes')
                successful={i for i,row in enumerate(report['cases']) if row.get('success') and row.get('full_success')}
                if set(meta['episodes'])!=successful:raise ValueError('Training episodes must be the complete physical successes')
                if archive['obs'].shape[1]!=98 or archive['actions'].shape[1]!=14:raise ValueError('Incompatible actor dimensions')
                if not np.isfinite(archive['obs']).all() or not np.isfinite(archive['actions']).all():raise ValueError('Nonfinite demonstrations')
            destination=folder/f'dataset_{i}.npz';shutil.copy2(path,destination);datasets.append(destination)
            provenance.append(dict(original=str(path),snapshot=str(destination),sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),episodes=meta['episodes']))
        status['datasets']=provenance;save()
        pose=args.output/'pose_bc';jaws=args.output/'pressure_bc'
        run('autonomy_main.py',['train','--skill','full','--datasets',*datasets,'--output',pose,
            '--architecture','linear_options','--latch_option_poses','--pose_entry_supervision','--exclude_joint_shortcuts',
            '--future_steps',20,'--updates',24000,'--batch_size',1024,'--seed',42], 'pose_training')
        run('autonomy_main.py',['fit_jaws','--skill','full','--policy',pose/'policy.pt','--datasets',*datasets,'--output',jaws,
            '--pressure_supervision','--pressure_min',.36,'--pressure_max',.40,'--updates',10000,'--batch_size',1024,'--seed',42], 'jaw_training')
        policy=jaws/'policy.pt';digest=hashlib.sha256(policy.read_bytes()).hexdigest();status.update(policy=str(policy),policy_sha256=digest);save()
        for label,count,seed,jitter,yaw in [('pilot',3,123,0,0),('validation_seed123',16,123,.25,1),('validation_seed9876',16,9876,.25,1)]:
            out=args.output/label
            run('autonomy_replay.py',['evaluate','--skill','full','--policy',policy,'--output',out,'--num_envs',count,
                '--episodes',count,'--episode_steps',10000,'--sources','L5','--targets','R2',
                '--jitter_mm',jitter,'--yaw_deg',yaw,'--seed',seed,'--headless'],label)
            report=json.loads((out/'evaluation.json').read_text())
            if report.get('policy_sha256')!=digest or hashlib.sha256(policy.read_bytes()).hexdigest()!=digest:raise ValueError('Evaluation weights changed')
            status[label]=dict(success=report['success'],episodes=report['episodes'],report=str(out/'evaluation.json'));save()
            if not native_full_gate(report,count,1. if label=='pilot' else .9):
                status.update(complete=True,stage='not_promoted',reason=f'{label} failed native full certification; inspect physical failures before another round');save();return 1
        status.update(complete=True,stage='certified',native_full_policy_verified=True);save()
        print(json.dumps(status,indent=2),flush=True);return 0
    except BaseException as exc:
        status.update(complete=False,error=f'{type(exc).__name__}: {exc}');save();raise

if __name__=='__main__':raise SystemExit(main())
