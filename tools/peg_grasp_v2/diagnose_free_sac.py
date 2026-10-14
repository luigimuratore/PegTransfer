"""Inspect a saved free SAC actor using its verified, unchanged physics sources."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();checkpoint=args.checkpoint.resolve();folder=checkpoint.parent/'sources'
    manifest=json.loads((checkpoint.parent/'manifest.json').read_text())
    for name,digest in manifest['source_hashes'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Source changed: {name}')
    if checkpoint.name=='final.zip' and hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=manifest['checkpoint_sha256']:
        raise ValueError('Weights changed')
    if checkpoint.name=='pretrained.zip':
        evidence=json.loads((checkpoint.parent/'pretrained_evaluation.json').read_text())
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=evidence['checkpoint_sha256']:
            raise ValueError('Pretrained weights changed')
    if args.steps<1:p.error('Positive steps required')
    sys.path.insert(0,str(folder));frozen=importlib.import_module('rl_main')
    frozen.REPO=REPO;frozen.HERE=folder
    def probe(model,env,episodes,realtime=False):
        import torch
        from rl_core import apply,rotvec,multiply,conjugate
        from geometry import contact_candidate
        obs=env.reset();rows=[];minimum=float('inf');peak=0.
        local=torch.tensor([contact_candidate(env.profile,'wall',2 if arm==1 else 0,.5,arm)['tip_local_m']
                            for arm in (1,2)],device=env.device,dtype=torch.float32)
        def sample(step,action):
            nonlocal minimum,peak
            s=env.state
            sites=apply(s['peg_q'][:,None].expand(-1,2,-1),local[None])+s['peg'][:,None]
            distance=(s['tip']-sites).norm(dim=-1)*1000
            minimum=min(minimum,float(distance[0,0]));force=s['forces'].norm(dim=-1)
            peak=max(peak,float(force.max()))
            if step%100:return
            row=dict(step=step,tip_m=s['tip'][0].cpu().tolist(),peg_m=s['peg'][0].cpu().tolist(),
                wall_grasp_distance_mm=distance[0].cpu().tolist(),jaws_rad=s['jaws'][0].cpu().tolist(),
                motor_actions=action[0].tolist(),reference_m=env.reference.p[0].cpu().tolist(),
                reference_error_mm=((env.reference.p-s['tip']).norm(dim=-1)*1000)[0].cpu().tolist(),
                rotation_reference_error_rad=rotvec(multiply(env.reference.q,conjugate(s['tool_q'])))[0].norm(dim=-1).cpu().tolist(),
                joint_positions=s['joints'][0].cpu().tolist(),finger_force_n=force[0].cpu().tolist(),
                measured_flags=env.progress.flags[0].cpu().tolist())
            rows.append(row)
            print(f'[FREE SAC PROBE] step={step} distance_mm={row["wall_grasp_distance_mm"]} '
                  f'jaw={row["jaws_rad"][0]} action_psm1={row["motor_actions"][:7]}',flush=True)
        action,_=model.predict(obs,deterministic=True);sample(0,action)
        for step in range(1,args.steps+1):
            obs,_,done,_=env.step(action)
            action,_=model.predict(obs,deterministic=True);sample(step,action)
            if done.any():break
        with (args.output/'policy_probe.trace.jsonl').open('x') as file:
            for row in rows:file.write(json.dumps(row)+'\n')
        result=dict(episodes=len(env.episodes),success=sum(r['success'] for r in env.episodes),full_success=0,
            complete=False,diagnostic_only=True,steps=step,min_wall_grasp_distance_mm=minimum,
            peak_force_n=peak,assisted_grasp=False,cases=env.episodes,trace=str(args.output/'policy_probe.trace.jsonl'))
        (args.output/'policy_probe.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    frozen.run_evaluation=probe
    sys.argv=[str(folder/'rl_main.py'),'evaluate','--checkpoint',str(checkpoint),'--skill','lift',
              '--num_envs','1','--episodes','1','--episode_steps',str(args.steps+1),'--headless','--output',str(args.output)]
    return frozen.main()

if __name__=='__main__':raise SystemExit(main())
