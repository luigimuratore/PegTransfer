"""Compare faster motion on matched resets, then expand same-post offsets/yaw.

Full assistance stays constant: speed, randomization and support reduction are
separate experiments. A changed actor must pass the previous distribution too.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from autonomy_constraints_gate import assisted_gate

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
LEVELS=((0.,0.),(.25,1.),(.5,2.),(.75,3.5),(1.,5.))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def distribution_gate(report,jitter,yaw,episodes=16,threshold=.9):
    if not assisted_gate(report,1.,episodes,threshold,'firm',True):return False
    config=report.get('reset_distribution',{})
    if (config.get('sources')!=['L5'] or config.get('targets')!=['R2']
        or config.get('jitter_mm')!=jitter or config.get('yaw_deg')!=yaw or config.get('reset_only') is not True):return False
    for row in report['cases']:
        reset=row.get('reset',{});offset=reset.get('jitter_mm',[])
        post=reset.get('source_post_center_m',[])
        if (reset.get('source')!='L5' or reset.get('target')!='R2' or len(offset)!=2
            or any(not math.isfinite(v) or abs(v)>jitter+1e-5 for v in offset)
            or not math.isfinite(reset.get('yaw_deg',float('nan'))) or abs(reset['yaw_deg'])>yaw+1e-5
            or len(post)!=3 or any(not math.isfinite(a) or abs(a-b)>1e-7 for a,b in zip(post,[0.,0.,.01]))
            or reset.get('reset_only_randomization') is not True or reset.get('board_and_posts_unchanged') is not True):return False
    return True

def speed_comparison(reference,candidate,min_gain=.02):
    """Reject any lost successful case; never compare only overall rates."""
    if reference.get('policy_sha256')!=candidate.get('policy_sha256') or reference.get('seed')!=candidate.get('seed'):return False
    a=reference.get('cases',[]);b=candidate.get('cases',[])
    if not a or len(a)!=len(b):return False
    baseline=[];faster=[]
    for old,new in zip(a,b):
        for key in ('source','target','jitter_mm','yaw_deg'):
            if old.get('reset',{}).get(key)!=new.get('reset',{}).get(key):return False
        if old.get('success'):
            if not new.get('success') or new.get('peak_force_n',float('inf'))>1:return False
            baseline.append(old['length']);faster.append(new['length'])
    return bool(baseline) and sum(faster)<=sum(baseline)*(1-min_gain)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--recovery_rounds',type=int,default=1)
    args=p.parse_args()
    if not args.device.startswith('cuda') or not 0<=args.recovery_rounds<=3:p.error('CUDA and 0..3 recovery rounds required')
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,args.output/'curriculum_source.py')
    shutil.copy2(HERE/'autonomy_constraints_gate.py',args.output/'gate_source.py')
    state=dict(complete=False,stage='starting',algorithm='Frozen BC speed comparison; gated position curriculum with limited BC recoveries',
        levels=[dict(jitter_mm=j,yaw_deg=y) for j,y in LEVELS],support_strength=1.,support_profile='firm',placement_support=True,
        native_full_policy_verified=False,results=[],selected_speed=dict(free=1.,contact=1.),
        gate='At least 15/16 per seed123 and9876; faster motion must retain every matched baseline success and reduce mean duration by at least2%',
        policy=str(args.policy),policy_sha256=sha(args.policy))
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        temp=args.output/'status.json.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(args.output/'status.json')
    def run(module,argv,label):
        state['stage']=label;save()
        with (args.output/(label+'.log')).open('w') as stream:
            child=subprocess.Popen([sys.executable,'-u',str(HERE/module),*map(str,argv)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save()
            try:code=child.wait()
            except BaseException:child.send_signal(signal.SIGINT);child.wait();raise
        state.pop('child_pid',None);save()
        if code not in (0,1):raise RuntimeError(f'{label} exited {code}')
        return code
    def evaluate(policy,label,jitter,yaw,seed,count=16,free=1.,contact=1.):
        folder=args.output/label
        code=run('autonomy_replay.py',['evaluate','--policy',policy,'--output',folder,'--skill','full',
            '--num_envs',min(count,16),'--episodes',count,'--episode_steps',10000,'--seed',seed,'--device',args.device,
            '--sources','L5','--targets','R2','--jitter_mm',jitter,'--yaw_deg',yaw,'--grasp_support',1,
            '--support_profile','firm','--placement_support','--free_speed_scale',free,'--contact_speed_scale',contact,'--headless'],label)
        report_path=folder/'evaluation.json';report=json.loads(report_path.read_text())
        close=json.loads((folder/'environment_close.json').read_text())
        if (close.get('environment_closed') is not True or close.get('evaluation_sha256')!=sha(report_path)
            or report.get('policy_sha256')!=sha(policy) or report.get('seed')!=seed):raise ValueError('Unverified physical report/cleanup/weights')
        speed=report.get('motion_reference',{})
        if speed.get('free_scale')!=free or speed.get('contact_scale')!=contact:raise ValueError('Speed differs from requested comparison')
        passed=distribution_gate(report,jitter,yaw,count,1. if count==3 else .9)
        if passed and code!=0:raise ValueError('Passed task must exit successfully')
        state['results'].append(dict(label=label,success=report['success'],episodes=count,passed=passed,
            policy_sha256=sha(policy),jitter_mm=jitter,yaw_deg=yaw,seed=seed,free_scale=free,contact_scale=contact,
            report=str(report_path),report_sha256=sha(report_path)));save()
        return report,passed
    try:
        save()
        # Freeze current code and retain exact checkpoint8 weights.
        policy=args.output/'runtime'/'policy.pt'
        run('autonomy_main.py',['repackage','--policy',args.policy,'--output',policy.parent,'--skill','full',
            '--implementation_change','Explicit measured speed comparison and same-post reset curriculum; original model unchanged'], 'freeze_runtime')
        if sha(policy)!=sha(args.policy):raise ValueError('Speed experiment changed neural weights')
        state['policy']=str(policy);save()
        pilot,ok=evaluate(policy,'baseline_fixed',0,0,123,3)
        if not ok:
            state.update(complete=True,stage='baseline_failed');save();return 1
        baseline={}
        for seed in (123,9876):
            baseline[seed],ok=evaluate(policy,f'baseline_seed{seed}',.25,1,seed)
            if not ok:
                state.update(complete=True,stage='baseline_randomized_failed',reason='Keep checkpoint8; do not accelerate an unstable reference');save();return 1
        state.update(last_passed_level=1,last_verified_policy=str(policy),last_verified_policy_sha256=sha(policy));save()
        selected=(1.,1.)
        selected_reports=baseline
        for index,(free,contact) in enumerate(((1.5,1.25),(1.5,1.))):
            _,ok=evaluate(policy,f'speed{index}_fixed',0,0,123,3,free,contact)
            if not ok:continue
            compared={};accepted=True
            for seed in (123,9876):
                report,ok=evaluate(policy,f'speed{index}_seed{seed}',.25,1,seed,16,free,contact)
                paired=ok and speed_comparison(baseline[seed],report)
                state['results'][-1]['matched_speed_accepted']=paired;save()
                compared[seed]=report
                if not paired:accepted=False;break
            if accepted:selected=(free,contact);selected_reports=compared;break
        state['selected_speed']=dict(free=selected[0],contact=selected[1]);state['last_passed_level']=1;save()
        state['speed_reference_reports']={str(k):dict(success=v['success'],episodes=v['episodes']) for k,v in selected_reports.items()}
        for level,(jitter,yaw) in enumerate(LEVELS[2:],2):
            promoted=False
            for round_index in range(args.recovery_rounds+1):
                reports=[];passed=True
                for seed in (123,9876):
                    report,ok=evaluate(policy,f'level{level}_round{round_index}_seed{seed}',jitter,yaw,seed,16,*selected)
                    reports.append(report);passed &= ok
                if passed:
                    promoted=True;state.update(last_passed_level=level,policy=str(policy),policy_sha256=sha(policy),
                        last_verified_policy=str(policy),last_verified_policy_sha256=sha(policy));save();break
                failed=[c for report in reports for c in report['cases'] if not c['success']]
                # An approach-only fit cannot fix an insertion/handover fault.
                if round_index==args.recovery_rounds or any(c.get('grasp_psm1') for c in failed):
                    state.update(complete=True,stage='randomization_gate_failed',failed_level=level,
                        reason='Do not widen reset range: inspect failed approach/handover/placement cases; retain last passed policy and distribution');save();return 1
                collection=args.output/f'level{level}_recovery{round_index}'
                run('autonomy_main.py',['collect','--policy',policy,'--output',collection,'--skill','full','--num_envs',16,
                    '--episodes',32,'--episode_steps',10000,'--seed',4242+level*100+round_index,'--device',args.device,
                    '--jitter_mm',jitter,'--yaw_deg',yaw,'--sources','L5','--targets','R2',
                    '--grasp_support',1,'--support_profile','firm','--placement_support','--headless'],f'level{level}_collect{round_index}')
                dataset=collection/'demonstrations.npz'
                if not dataset.exists():raise ValueError('No successful recovery dataset')
                candidate=args.output/f'level{level}_fit{round_index}'
                run('autonomy_position_fit.py',['--policy',policy,'--dataset',dataset,'--output',candidate,
                    '--device',args.device],f'level{level}_fit{round_index}')
                candidate_policy=candidate/'policy.pt'
                # Keep the last accepted model until fixed and previous-level checks pass.
                _,fixed=evaluate(candidate_policy,f'level{level}_regression{round_index}_fixed',0,0,123,3,*selected)
                regression=fixed
                for seed in (123,9876):
                    _,ok=evaluate(candidate_policy,f'level{level}_regression{round_index}_seed{seed}',*LEVELS[level-1],seed,16,*selected)
                    regression &= ok
                if not regression:
                    state.update(complete=True,stage='recovery_not_promoted',reason='Fine-tuned approach degraded previous conditions');save();return 1
                policy=candidate_policy
            if not promoted:raise ValueError('Level ended without verification')
        state.update(complete=True,stage='same_post_curriculum_verified',policy=str(policy),policy_sha256=sha(policy),
            verified_conditions=dict(source='L5',target='R2',jitter_mm=1.,yaw_deg=5.),
            reason='Assisted full task only; support reduction is a separate subsequent curriculum');save();return 0
    except BaseException as error:
        state.update(complete=False,error=f'{type(error).__name__}: {error}');save();raise

if __name__=='__main__':raise SystemExit(main())
