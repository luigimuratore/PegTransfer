"""CUDA verification on measured full episodes plus motor safety/cause checks."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from full_learning import FullMotorPolicy,load_data,observe,decode_tensor,STAGES,OBS_DIM,speeds
from full_play import verify_model

def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,default=Path('logs/grasp_v2/full_bc1/policy.npz'))
    p.add_argument('--output',type=Path)
    args=p.parse_args();manifest,_=verify_model(args.policy)
    policy=FullMotorPolicy(args.policy,'cuda:0');data,meta=load_data(Path(manifest['demonstration_path']))
    held=data['episode']==manifest['validation_episode']
    assert not set(manifest['training_episodes'])&{manifest['validation_episode']}
    indices=np.r_[*[np.flatnonzero(held&(data['stage']==i))[:8] for i in range(len(STAGES))]]
    raw=torch.as_tensor(data['observations'][indices],device=policy.device)
    with torch.inference_mode():
        x=(raw-policy.mean)/policy.std
        for i,(w,b) in enumerate(policy.layers):
            x=torch.nn.functional.linear(x,w,b);x=x.tanh() if i==2 else x.relu()
        batch=decode_tensor(raw,x,policy.metadata['parameters']).cpu().numpy()
    individual=np.stack([policy.predict(obs) for obs in data['observations'][indices]])
    difference=np.abs(batch-individual)
    worst=np.unravel_index(difference.argmax(),difference.shape)
    # TF32 matmul may round batched and single calls differently. Bound the
    # actual actuator difference: 25 nm translation / 0.1 microrad jaw.
    translation_difference=difference[:,:3]*.01*speeds(meta['parameters'])[data['stage'][indices],None]
    assert translation_difference.max()<2.5e-8 and (difference[:,3:]*.0015).max()<1e-7,dict(max_difference=float(difference.max()),
        stage=STAGES[data['stage'][indices[worst[0]]]],batch=batch[worst[0]].tolist(),single=individual[worst[0]].tolist())
    # A blocked arm cannot accumulate an unlimited reference lead. Even at
    # maximum learned gains, the near-contact command stays within .5 mm.
    test=torch.zeros((4,OBS_DIM),device=policy.device)
    test[:,54+STAGES.index('receiver_close')]=1.;test[:,27]=2.
    test[:,51]=.2;test[:,24]=.2;test[:,74:76]=.2
    test[1,43]=.02 # Only receiver finger one touches: pause closure to center.
    test[2,43]=.5;test[2,46]=-.5 # Excess pressure must open, not squeeze harder.
    with torch.inference_mode():commands=decode_tensor(test,torch.ones((4,5),device=policy.device),meta['parameters']).cpu().numpy()
    assert commands[0,4]<0 and commands[1,4]==0 and commands[2,4]>0
    assert np.linalg.norm(commands[:,:3]*.01*meta['parameters']['contact_speed'],axis=1).max()<=.00050001
    # The donor has released: subsequent receiver motion must retain its
    # opening, despite the zero-contact pressure input and arbitrary network.
    test[:]=0;test[:,54+STAGES.index('receiver_transport')]=1;test[:,24]=.8;test[:,51]=.2
    with torch.inference_mode():commands=decode_tensor(test,-torch.ones((4,5),device=policy.device),meta['parameters']).cpu().numpy()
    assert np.all(commands[:,3]==0)
    # Observation construction reads current telemetry only. Future, clock
    # and reward annotations cannot change inference.
    report=json.loads(Path(meta['report']).read_text())
    with Path(report['trace']).open() as f:row=json.loads(next(f))
    rows=row['pre_action_state'];refs=[r['tool_tip_w_m'] for r in rows];quats=[r['tool_quaternion_wxyz'] for r in rows]
    jaw=[meta['parameters']['donor_open'],meta['parameters']['receiver_open']]
    before=observe(rows,'settle',refs,quats,refs,jaw,jaw)
    altered=copy.deepcopy(rows)
    for arm in altered:arm.update(step=10000,reward=999,phase_age=123,next_state={'tool_tip_w_m':[10,10,10]})
    after=observe(altered,'settle',refs,quats,refs,jaw,jaw)
    assert np.array_equal(before,after)
    result=dict(passed=True,device=str(policy.device),gpu_name=torch.cuda.get_device_name(policy.device),
        checks=['20 phases on held-out episode','saved CUDA model batch/single inference parity',
            'contact approach reference bounded','unbalanced receiver contact pauses closure',
            'high pressure opens jaws','released inactive donor stays open','clock/reward/future annotations ignored',
            'model and preserved scene hashes verified'],
        max_batch_single_action_difference=float(np.abs(batch-individual).max()),
        held_out_action_mse=manifest['validation']['normalized_action_mse'])
    if args.output:
        with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
