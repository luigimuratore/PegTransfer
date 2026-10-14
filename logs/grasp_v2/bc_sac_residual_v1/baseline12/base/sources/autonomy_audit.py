"""CUDA offline audit; compares checkpoints on HELD-OUT teacher observations."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from autonomy_learning import load_policy,future_poses,future_actions

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--policies',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4)
    with np.load(args.dataset,allow_pickle=False) as archive:
        data={k:torch.as_tensor(archive[k],device='cuda:0') for k in ('obs','actions','episode','stage','validation')}
    rows=[]
    for path in args.policies:
        net=load_policy(path,'cuda:0');cfg=json.loads((path.parent/'training.json').read_text())
        mask=data['validation'];ids=mask.nonzero().flatten()
        target=(future_poses(data,cfg.get('future_pose_steps',0),cfg.get('pose_supervision')=='measured phase endpoints')
            if hasattr(net,'poses') else future_actions(data,cfg.get('future_pose_steps',0)))
        with torch.inference_mode():
            if hasattr(net,'outputs'):
                # Offline state error is NOT a persistent-option rollout.
                # Do not accidentally initialize every sampled state in the
                # initial option or mix unrelated validation episodes as history.
                logits,heads=net.outputs(data['obs'][ids])
                prediction=heads[torch.arange(len(ids),device='cuda:0'),logits.argmax(-1)]
            else:prediction=net.poses(data['obs'][ids]) if hasattr(net,'poses') else net(data['obs'][ids])
            stage_errors=[]
            for stage in data['stage'][ids].unique():
                chosen=data['stage'][ids]==stage
                error=(prediction[chosen]-target[ids][chosen]).square().mean(0).sqrt()
                stage_errors.append(dict(stage=int(stage),rmse=error.cpu().tolist()))
            first=(data['episode']==data['episode'][ids][0]).nonzero().flatten()[0]
            sample=data['obs'][first:first+1]
            if hasattr(net,'reset'):net.reset()
            rows.append(dict(policy=str(path),stage_errors=stage_errors,first_observation=sample[0].cpu().tolist(),
                offline_selection='Instantaneous gate only; persistent option/pose memory requires physical rollout' if hasattr(net,'outputs') else 'Direct prediction',
                first_actions=net(sample)[0].cpu().tolist(),
                first_pose=net.poses(sample)[0].cpu().tolist() if hasattr(net,'poses') else None,
                first_target=target[first].cpu().tolist()))
    args.output.write_text(json.dumps(dict(offline_only=True,teacher_state_audit=True,results=rows),indent=2)+'\n')
    print(args.output)

if __name__=='__main__':main()
