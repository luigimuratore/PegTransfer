"""Compare clearance-based transit speeds on identical V48 resets and weights."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
from autonomy_motion_curriculum import distribution_gate,speed_comparison,sha

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
TRIALS=((2.,2.),(3.,3.),(4.,4.))


def verified_report(path,digest,seed,jitter,yaw,count):
    path=Path(path)
    report=json.loads(path.read_text())
    close=json.loads((path.parent/'environment_close.json').read_text())
    if (close.get('environment_closed') is not True or close.get('evaluation_complete') is not True
        or close.get('evaluation_sha256')!=sha(path) or report.get('policy_sha256')!=digest
        or report.get('seed')!=seed):raise ValueError('Incomplete or incompatible physical report/cleanup')
    return report,distribution_gate(report,jitter,yaw,count,1. if count==3 else .9)


def reference_reports(validation,policy):
    status=json.loads((validation/'status.json').read_text());digest=sha(policy)
    if (status.get('complete') is not True or status.get('stage')!='assisted_receiver_candidate_validated'
        or status.get('candidate_policy_sha256')!=digest):raise ValueError('Completed V48 validation required')
    result={}
    for label,seed,jitter,yaw,count in [('fixed_pilot',123,0.,0.,3),('fresh_seed2468',2468,.5,2.,16),('fresh_seed54321',54321,.5,2.,16)]:
        matches=[row for row in status['results'] if row['label']==label]
        if len(matches)!=1:raise ValueError('Missing reference condition')
        row=matches[0];path=ROOT/row['report']
        if sha(path)!=row['report_sha256']:raise ValueError('Reference report changed')
        report,ok=verified_report(path,digest,seed,jitter,yaw,count)
        if not ok or report['motion_reference']['free_scale']!=1 or report['motion_reference']['contact_scale']!=1:
            raise ValueError('Reference must pass at original speeds')
        result[label]=report
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--motion_profile',choices=('clearance','travel'),default='travel')
    args=p.parse_args();baseline=reference_reports(args.validation,args.policy)
    args.output.mkdir(parents=True,exist_ok=False)
    state=dict(complete=False,stage='freeze_runtime',algorithm='Frozen BC transit speed comparison; no training',
        policy_sha256=sha(args.policy),parent_policy=str(args.policy),results=[],
        assisted_grasp=True,native_full_policy_verified=False,weights_changed=False,
        interaction_speed_mm_s=3.,jaw_speeds_and_hold_times_changed=False,
        selected=None,motion_profile=args.motion_profile,
        gate='3/3 fixed; at least15/16 per seed2468 and54321; retain every baseline successful reset and reduce matched mean simulated duration by at least2%')
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        tmp=args.output/'status.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(args.output/'status.json')
    def run(module,argv,label):
        state['stage']=label;save()
        with (args.output/f'{label}.log').open('w') as stream:
            child=subprocess.Popen([sys.executable,'-u',str(HERE/module),*map(str,argv)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save()
            try:code=child.wait()
            except BaseException:child.send_signal(signal.SIGINT);child.wait();raise
        state.pop('child_pid',None);save()
        if code not in (0,1):raise RuntimeError(f'{label} exited {code}')
        return code
    try:
        save();policy=args.output/'runtime/policy.pt'
        run('autonomy_main.py',['repackage','--policy',args.policy,'--output',policy.parent,'--skill','full',
            '--implementation_change','Opt-in transit translations; travel preserves original rotation caps and approach minimum speed; weights/jaws/progress/physics unchanged'], 'freeze_runtime')
        if sha(policy)!=state['policy_sha256']:raise ValueError('Neural weights changed')
        state['runtime_policy']=str(policy);save()
        def evaluate(label,seed,count,jitter,yaw,profile,free=1.,carry=1.):
            out=args.output/label
            code=run('autonomy_replay.py',['evaluate','--policy',policy,'--output',out,'--skill','full',
                '--num_envs',count,'--episodes',count,'--episode_steps',10000,'--seed',seed,'--sources','L5','--targets','R2',
                '--jitter_mm',jitter,'--yaw_deg',yaw,'--grasp_support',1.,'--support_profile','firm','--placement_support',
                '--motion_profile',profile,'--free_speed_scale',free,'--transport_speed_scale',carry,'--headless'],label)
            path=out/'evaluation.json';report,passed=verified_report(path,state['policy_sha256'],seed,jitter,yaw,count)
            speed=report['motion_reference']
            if (speed.get('profile','contact')!=profile or speed['free_scale']!=free or speed['contact_scale']!=1
                or speed.get('transport_scale',1)!=carry):raise ValueError('Requested and measured speed configuration differ')
            if passed and code!=0:raise ValueError('Successful evaluation exited unsuccessfully')
            row=dict(label=label,seed=seed,success=report['success'],episodes=count,physical_gate_passed=passed,
                motion_profile=profile,free_speed_scale=free,transport_speed_scale=carry,
                report=str(path),report_sha256=sha(path))
            state['results'].append(row);save();return report,passed,row
        _,passed,_=evaluate('baseline_runtime_fixed',123,3,0.,0.,'contact')
        if not passed:
            state.update(complete=True,stage='runtime_regression_failed');save();return 1
        for index,(free,carry) in enumerate(TRIALS):
            label=f'transit{index}'
            report,passed,row=evaluate(f'{label}_fixed',123,3,0.,0.,args.motion_profile,free,carry)
            paired=passed and speed_comparison(baseline['fixed_pilot'],report)
            row['matched_speed_accepted']=paired;save()
            if not paired:continue
            evidence=[dict(label=row['label'],report=row['report'],report_sha256=row['report_sha256'])]
            accepted=True;durations=[]
            for seed in (2468,54321):
                report,passed,row=evaluate(f'{label}_seed{seed}',seed,16,.5,2.,args.motion_profile,free,carry)
                reference=baseline[f'fresh_seed{seed}']
                paired=passed and speed_comparison(reference,report)
                row['matched_speed_accepted']=paired
                successful=[i for i,c in enumerate(reference['cases']) if c['success']]
                before=sum(reference['cases'][i]['length'] for i in successful)
                after=sum(report['cases'][i]['length'] for i in successful)
                row['matched_duration_gain_percent']=100*(1-after/before)
                durations.append(after/len(successful)*.01)
                evidence.append(dict(label=row['label'],report=row['report'],report_sha256=row['report_sha256']))
                save()
                if not paired:accepted=False;break
            if accepted:
                mean=sum(durations)/len(durations)
                if state['selected'] is None or mean<state['selected']['matched_mean_seconds']:
                    state['selected']=dict(policy=str(policy),policy_sha256=state['policy_sha256'],
                        motion_profile=args.motion_profile,free_speed_scale=free,transport_speed_scale=carry,
                        matched_mean_seconds=mean,reports=evidence)
                save()
        state.update(complete=True,stage='transit_speed_validated' if state['selected'] else 'speed_not_promoted',
            fallback_policy=str(args.policy),reason='Original V48 retained; only fully matched faster trials are selectable')
        save();return 0 if state['selected'] else 1
    except BaseException as error:
        state.update(complete=False,stage='interrupted' if isinstance(error,KeyboardInterrupt) else 'workflow_error',
            error=f'{type(error).__name__}: {error}');save();raise


if __name__=='__main__':raise SystemExit(main())
