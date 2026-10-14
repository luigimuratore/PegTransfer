"""Run bounded BC/DAgger rounds; promote only an independently evaluated actor."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def pickup_gate(report,sources):
    """Check the whole allocation AND every source post before promotion."""
    cases=report.get('cases',[])
    if not report.get('complete') or report.get('teacher_at_inference') is not False or report.get('assisted_grasp') is not False:return False
    if len(cases)!=report.get('episodes') or len(cases)<32:return False
    if sum(bool(c['success']) for c in cases)!=report.get('success'):return False
    if report['success']/len(cases)<.9:return False
    for source in sources:
        rows=[c for c in cases if c.get('reset',{}).get('source')==source]
        if len(rows)<8 or sum(bool(c['success']) for c in rows)/len(rows)<.9:return False
    return True

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--dataset',type=Path,nargs='+')
    p.add_argument('--wait_dataset',type=Path,help='Append a collection still running; wait at most 30 minutes')
    p.add_argument('--initial_policy',type=Path,help='Evaluate an existing preserved lift candidate before any new training')
    p.add_argument('--recovery_episodes',type=int,default=48)
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--num_envs',type=int,default=16)
    p.add_argument('--updates',type=int,default=10000);p.add_argument('--device',default='cuda:0')
    p.add_argument('--architecture',choices=('mlp','gru','pose','mixture','linear_options'),default='linear_options')
    p.add_argument('--future_steps',type=int,default=20)
    p.add_argument('--pressure_bc',action='store_true',help='After movement BC, fit learned pressure decisions on native training data')
    p.add_argument('--jaw_updates',type=int,default=6000)
    p.add_argument('--sources',nargs='+',choices=('L5','L6'),default=['L5'])
    p.add_argument('--targets',nargs='+',choices=('R1','R2'),default=['R2'])
    p.add_argument('--jitter_mm',type=float,default=.5);p.add_argument('--yaw_deg',type=float,default=2.)
    p.add_argument('--pressure_min',type=float,default=.36);p.add_argument('--pressure_max',type=float,default=.40)
    args=p.parse_args()
    if args.rounds<1 or args.num_envs<1 or args.recovery_episodes<1 or args.jaw_updates<1:p.error('Positive rounds and environments required')
    if args.pressure_bc and args.architecture not in ('mixture','linear_options'):p.error('Pressure BC requires option architecture')
    if args.initial_policy and not args.initial_policy.exists():p.error('initial_policy must exist')
    if not 0<=args.jitter_mm<=1 or not 0<=args.yaw_deg<=5:p.error('First curriculum permits jitter 0..1 mm and yaw 0..5 deg')
    if not .1<=args.pressure_min<args.pressure_max<=.8:p.error('Invalid pressure label band')
    args.output.mkdir(parents=True,exist_ok=False)
    status=dict(phase='pickup',complete=False,autonomous_pickup_verified=False,full_task_verified=False,rounds=[],
        conditions=dict(sources=args.sources,targets=args.targets,jitter_mm=args.jitter_mm,yaw_deg=args.yaw_deg),
        gate='At least 90% overall and on each source; at least 8 episodes per source; two seeds, same weights')
    write(args.output/'status.json',status)
    def run(module,argv,folder):
        command=[sys.executable,'-u',str(HERE/module),*map(str,argv)]
        print('[CURRICULUM] '+' '.join(command),flush=True)
        code=subprocess.call(command,cwd=REPO)
        error=folder/'error.json'
        if error.exists():raise RuntimeError(error.read_text())
        if code not in (0,1):raise RuntimeError(f'Command exited {code}')
        return code
    conditions=['--sources',*args.sources,'--targets',*args.targets,'--jitter_mm',args.jitter_mm,'--yaw_deg',args.yaw_deg]
    try:
        datasets=[]
        if args.dataset:datasets.extend(path.resolve() for path in args.dataset)
        else:
            folder=args.output/'teacher'
            run('autonomy_main.py',['collect','--output',folder,'--num_envs',args.num_envs,'--episodes',48,
                '--headless',*conditions,'--device',args.device],folder)
            datasets.append(folder/'demonstrations.npz')
        if args.wait_dataset:
            deadline=time.monotonic()+1800
            while not args.wait_dataset.exists():
                if time.monotonic()>deadline:raise RuntimeError('Timed out waiting for recovery dataset')
                if (args.wait_dataset.parent/'error.json').exists():raise RuntimeError('Recovery collection failed')
                print('[CURRICULUM] Waiting for complete recovery dataset',flush=True);time.sleep(30)
            datasets.append(args.wait_dataset.resolve())
        for index in range(args.rounds):
            if index==0 and args.initial_policy:policy=args.initial_policy.resolve()
            else:
                folder=args.output/f'bc_round{index}'
                run('autonomy_main.py',['train','--output',folder,'--datasets',*datasets,'--updates',args.updates,
                    '--device',args.device,'--seed',42+index,'--architecture',args.architecture,'--future_steps',args.future_steps,
                    *(['--latch_option_poses'] if args.architecture in ('mixture','linear_options') else [])],folder)
                policy=folder/'policy.pt'
                if args.pressure_bc:
                    jaw_folder=args.output/f'jaw_bc_round{index}'
                    run('autonomy_main.py',['fit_jaws','--policy',policy,'--output',jaw_folder,'--datasets',*datasets,
                        '--pressure_supervision','--pressure_min',args.pressure_min,'--pressure_max',args.pressure_max,
                        '--updates',args.jaw_updates,'--seed',42+index,'--device',args.device],jaw_folder)
                    policy=jaw_folder/'policy.pt'
            eval_folder=args.output/f'eval_round{index}'
            run('autonomy_replay.py',['evaluate','--policy',policy,'--output',eval_folder,'--num_envs',args.num_envs,
                '--episodes',32,'--headless','--seed',123+index*100,*conditions,'--device',args.device],eval_folder)
            report=json.loads((eval_folder/'evaluation.json').read_text())
            row=dict(round=index,policy=str(policy),evaluation=str(eval_folder/'evaluation.json'),
                success=report['success'],episodes=report['episodes'])
            status['rounds'].append(row);write(args.output/'status.json',status)
            if pickup_gate(report,args.sources):
                # A second seed checks the same exact weights before promotion.
                confirmation=args.output/f'confirmation_round{index}'
                run('autonomy_replay.py',['evaluate','--policy',policy,'--output',confirmation,'--num_envs',args.num_envs,
                    '--episodes',32,'--headless','--seed',9876,*conditions,'--device',args.device],confirmation)
                confirm=json.loads((confirmation/'evaluation.json').read_text())
                if pickup_gate(confirm,args.sources) and confirm['policy_sha256']==report['policy_sha256']:
                    status.update(complete=True,autonomous_pickup_verified=True,policy=str(policy),
                        policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),confirmation=str(confirmation/'evaluation.json'),
                        next_phase='Expand valid source posts, then measured unassisted handover. Full task not yet certified.')
                    write(args.output/'status.json',status);print(json.dumps(status,indent=2),flush=True);return 0
            if index+1<args.rounds:
                correction=args.output/f'dagger_round{index+1}'
                run('autonomy_main.py',['collect','--output',correction,'--num_envs',args.num_envs,'--episodes',args.recovery_episodes,
                    '--policy',policy,'--learner_fraction',.1 if index==0 else .25,'--learner_until_stage',3,'--headless',
                    *conditions,'--seed',1042+index*1000,'--device',args.device],correction)
                datasets.append(correction/'demonstrations.npz')
                recovery=json.loads((correction/'evaluation.json').read_text())
                if recovery['success']<3:
                    status.update(complete=True,reason='Too few successful recovery episodes; diagnose before another training round.')
                    write(args.output/'status.json',status);return 1
        status.update(complete=True,reason='Pickup gate not reached; preserve failed models, diagnose before extending curriculum.')
        write(args.output/'status.json',status);return 1
    except BaseException as exc:
        status.update(complete=False,error=f'{type(exc).__name__}: {exc}')
        write(args.output/'status.json',status);raise

if __name__=='__main__':raise SystemExit(main())
