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

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--dataset',type=Path,nargs='+')
    p.add_argument('--wait_dataset',type=Path,help='Append a collection still running; wait at most 30 minutes')
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--num_envs',type=int,default=16)
    p.add_argument('--updates',type=int,default=10000);p.add_argument('--device',default='cuda:0')
    p.add_argument('--architecture',choices=('mlp','gru','pose','mixture','linear_options'),default='linear_options')
    p.add_argument('--future_steps',type=int,default=20)
    args=p.parse_args()
    if args.rounds<1 or args.num_envs<1:p.error('Positive rounds and environments required')
    args.output.mkdir(parents=True,exist_ok=False)
    status=dict(phase='pickup',complete=False,autonomous_pickup_verified=False,full_task_verified=False,rounds=[])
    write(args.output/'status.json',status)
    def run(module,argv,folder):
        command=[sys.executable,'-u',str(HERE/module),*map(str,argv)]
        print('[CURRICULUM] '+' '.join(command),flush=True)
        code=subprocess.call(command,cwd=REPO)
        error=folder/'error.json'
        if error.exists():raise RuntimeError(error.read_text())
        if code not in (0,1):raise RuntimeError(f'Command exited {code}')
        return code
    try:
        datasets=[]
        if args.dataset:datasets.extend(path.resolve() for path in args.dataset)
        else:
            folder=args.output/'teacher'
            run('autonomy_main.py',['collect','--output',folder,'--num_envs',args.num_envs,'--episodes',48,
                '--headless','--jitter_mm',.5,'--yaw_deg',2,'--device',args.device],folder)
            datasets.append(folder/'demonstrations.npz')
        if args.wait_dataset:
            deadline=time.monotonic()+1800
            while not args.wait_dataset.exists():
                if time.monotonic()>deadline:raise RuntimeError('Timed out waiting for recovery dataset')
                if (args.wait_dataset.parent/'error.json').exists():raise RuntimeError('Recovery collection failed')
                print('[CURRICULUM] Waiting for complete recovery dataset',flush=True);time.sleep(30)
            datasets.append(args.wait_dataset.resolve())
        for index in range(args.rounds):
            folder=args.output/f'bc_round{index}'
            run('autonomy_main.py',['train','--output',folder,'--datasets',*datasets,'--updates',args.updates,
                '--device',args.device,'--seed',42+index,'--architecture',args.architecture,'--future_steps',args.future_steps],folder)
            policy=folder/'policy.pt';eval_folder=args.output/f'eval_round{index}'
            run('autonomy_replay.py',['evaluate','--policy',policy,'--output',eval_folder,'--num_envs',args.num_envs,
                '--episodes',32,'--headless','--seed',123+index*100,'--jitter_mm',.5,'--yaw_deg',2,'--device',args.device],eval_folder)
            report=json.loads((eval_folder/'evaluation.json').read_text())
            row=dict(round=index,policy=str(policy),evaluation=str(eval_folder/'evaluation.json'),
                success=report['success'],episodes=report['episodes'])
            status['rounds'].append(row);write(args.output/'status.json',status)
            if report['complete'] and not report['teacher_at_inference'] and not report['assisted_grasp'] and report['success']/report['episodes']>=.9:
                # A second seed checks the same exact weights before promotion.
                confirmation=args.output/f'confirmation_round{index}'
                run('autonomy_replay.py',['evaluate','--policy',policy,'--output',confirmation,'--num_envs',args.num_envs,
                    '--episodes',32,'--headless','--seed',9876,'--jitter_mm',.5,'--yaw_deg',2,'--device',args.device],confirmation)
                confirm=json.loads((confirmation/'evaluation.json').read_text())
                if confirm['complete'] and confirm['success']/confirm['episodes']>=.9:
                    status.update(complete=True,autonomous_pickup_verified=True,policy=str(policy),
                        policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),confirmation=str(confirmation/'evaluation.json'),
                        next_phase='Expand valid source posts, then measured unassisted handover. Full task not yet certified.')
                    write(args.output/'status.json',status);print(json.dumps(status,indent=2),flush=True);return 0
            if index+1<args.rounds:
                correction=args.output/f'dagger_round{index+1}'
                run('autonomy_main.py',['collect','--output',correction,'--num_envs',args.num_envs,'--episodes',48,
                    '--policy',policy,'--learner_fraction',.1 if index==0 else .25,'--learner_until_stage',3,'--headless',
                    '--jitter_mm',.5,'--yaw_deg',2,'--seed',1042+index*1000,'--device',args.device],correction)
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
