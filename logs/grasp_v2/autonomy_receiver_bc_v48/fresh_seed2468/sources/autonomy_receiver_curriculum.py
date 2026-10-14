"""Wait for assisted recoveries, fit receiver jaws, gate physical regressions."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from autonomy_main import snapshot
from autonomy_motion_curriculum import distribution_gate,sha

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,required=True);parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--wait_minutes',type=float,default=45)
    args=parser.parse_args()
    if args.wait_minutes<=0:parser.error('Positive waiting limit required')
    args.output.mkdir(parents=True,exist_ok=False)
    frozen=args.output/'workflow';hashes=snapshot(frozen)
    state=dict(complete=False,stage='waiting_for_complete_recoveries',algorithm='Assisted receiver-only Behavior Cloning; not SAC',
        parent_policy=str(args.policy),parent_policy_sha256=sha(args.policy),dataset=str(args.dataset),
        original_motor_pickup_router_jaw_parameters_protected=True,results=[],assisted_grasp=True,native_full_policy_verified=False,
        validation='Fixed3/3; previous.25mm/1deg on123+9876; .5mm/2deg on fresh2468+54321, at least15/16 each',
        workflow_source_hashes=hashes)
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        temporary=args.output/'status.json.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(args.output/'status.json')
    def run(module,argv,label):
        # Never train from silently edited modules during the waiting period.
        for name,digest in hashes.items():
            if sha(HERE/name)!=digest:raise ValueError(f'Workflow source changed: {name}; use a new experiment')
        state['stage']=label;save()
        with (args.output/(label+'.log')).open('w') as log:
            child=subprocess.Popen([sys.executable,'-u',str(HERE/module),*map(str,argv)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save()
            try:code=child.wait()
            except BaseException:child.send_signal(signal.SIGINT);child.wait();raise
        state.pop('child_pid',None);save()
        if code not in (0,1):raise RuntimeError(f'{label} exited {code}')
        return code
    try:
        save();deadline=time.monotonic()+60*args.wait_minutes
        while not (args.dataset.parent/'environment_close.json').exists():
            error=args.dataset.parent/'error.json'
            if error.exists():raise RuntimeError(error.read_text())
            if time.monotonic()>deadline:raise TimeoutError('Recovery collection has not closed')
            time.sleep(10)
        if not args.dataset.is_file():raise ValueError('No complete successful recovery dataset')
        candidate=args.output/'receiver_bc'
        code=run('autonomy_receiver_fit.py',['--policy',args.policy,'--dataset',args.dataset,'--output',candidate],'receiver_bc_training')
        if code:raise RuntimeError('Receiver BC did not finish')
        policy=candidate/'policy.pt';digest=sha(policy)
        state.update(candidate_policy=str(policy),candidate_policy_sha256=digest);save()
        conditions=[('fixed_pilot',3,123,0.,0.),('previous_seed123',16,123,.25,1.),('previous_seed9876',16,9876,.25,1.),
                    ('fresh_seed2468',16,2468,.5,2.),('fresh_seed54321',16,54321,.5,2.)]
        for label,count,seed,jitter,yaw in conditions:
            out=args.output/label
            code=run('autonomy_replay.py',['evaluate','--policy',policy,'--output',out,'--skill','full',
                '--num_envs',count,'--episodes',count,'--episode_steps',10000,'--sources','L5','--targets','R2',
                '--jitter_mm',jitter,'--yaw_deg',yaw,'--seed',seed,'--grasp_support',1,'--support_profile','firm',
                '--placement_support','--headless'],label)
            report_path=out/'evaluation.json';report=json.loads(report_path.read_text());close=json.loads((out/'environment_close.json').read_text())
            if (close.get('environment_closed') is not True or close.get('evaluation_sha256')!=sha(report_path)
                or report.get('policy_sha256')!=digest or sha(policy)!=digest or report.get('seed')!=seed):
                raise ValueError('Evaluation hash/seed/clean closure mismatch')
            motion=report.get('motion_reference',{})
            if motion.get('free_scale')!=1 or motion.get('contact_scale')!=1:raise ValueError('Original speed required')
            passed=distribution_gate(report,jitter,yaw,count,1. if count==3 else 15/16)
            if passed and code:raise ValueError('Passed evaluation exited unsuccessfully')
            state['results'].append(dict(label=label,seed=seed,jitter_mm=jitter,yaw_deg=yaw,
                success=report['success'],episodes=count,passed=passed,report=str(report_path),report_sha256=sha(report_path)));save()
            if not passed:
                state.update(complete=True,stage='not_promoted',reason=f'{label} failed; checkpoint8 and V42 accepted settings remain the reference');save();return 1
        state.update(complete=True,stage='assisted_receiver_candidate_validated',validated_assisted_policy=str(policy),
            reason='Prior conditions and two fresh seeds passed; no automatic native or higher-randomization claim');save();return 0
    except BaseException as exc:
        state.update(complete=False,stage='workflow_error',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':raise SystemExit(main())
