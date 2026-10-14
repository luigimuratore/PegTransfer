"""Convert verified motor traces to free SAC actions, with complete-episode holdout."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from rl_core import *

REPO=Path(__file__).resolve().parents[2]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def trace_state(rows,previous,device):
    t=lambda x:torch.tensor(x,dtype=torch.float32,device=device)
    peg=t([rows[0]['peg_root_w_m']]);pegq=t([rows[0]['peg_quaternion_wxyz']])
    linvel=torch.zeros_like(peg);angvel=torch.zeros_like(peg)
    if previous is not None:
        linvel=(peg-t([previous[0]['peg_root_w_m']]))/DT
        angvel=rotvec(multiply(pegq,conjugate(t([previous[0]['peg_quaternion_wxyz']]))))/DT
    return dict(peg=peg,peg_q=pegq,linvel=linvel,angvel=angvel,
        tip=t([[r['tool_tip_w_m'] for r in rows]]),tool_q=t([[r['tool_quaternion_wxyz'] for r in rows]]),
        joints=t([[[r['joint_positions'][j] for j in JOINTS] for r in rows]]),
        joint_vel=t([[[r['joint_velocities'][j] for j in JOINTS] for r in rows]]),
        jaws=t([[r['jaw_rad'] for r in rows]]),forces=t([[r['finger_force_vectors_n'] for r in rows]]),
        contact_valid=torch.tensor([[[p is not None for p in r['finger_contact_points_w_m']] for r in rows]],device=device),
        relative=t([[r['peg_in_tool_m'] for r in rows]]),relative_q=t([[r['peg_quat_in_tool'] for r in rows]]),
        bottom=t([rows[0]['peg_bottom_m']]),speed=t([rows[0]['peg_speed_m_s']]),
        source=t([[0.,0.,.010]]),target=t([[.042,.015,.010]]))

def export(output,report=None):
    report=Path(report or REPO/'logs/grasp_v2/full_assisted_validation3_2026-10-05.json').resolve()
    from replay_full import verify
    manifest,bundle=verify(REPO/'milestones/checkpoint3-full-assisted.json')
    evidence=json.loads(report.read_text())
    if not evidence['complete'] or evidence['full_success']!=evidence['episodes'] or evidence.get('error'):
        raise ValueError('Successful complete audited episodes required')
    trace=REPO/evidence['trace'];profile=json.loads((bundle/'tools/peg_grasp_v2/peg_profile.json').read_text())
    if digest(report)!='54aadb6228fcf720af016e849d78198d2c4e650c8f0757d4450f1e55814198fe' or digest(trace)!='76f021cb11e94bff2b98e5200aa559f4ae9d20c8fc274b446a67dcaa23c37ff8':
        raise ValueError('This exporter requires the preserved checkpoint3 teacher trace')
    arrays={key:[] for key in ('obs','actions','next_obs','rewards','lift_rewards','done','episode','bc_valid','unassisted','lift_done','handover_done')}
    episode=None;previous=None;count=0
    # Parsing JSON and serial export are host work; simulation, replay, actor
    # and critic training run on CUDA. Never move physics into this exporter.
    with trace.open() as file:
        for line in file:
            row=json.loads(line)
            if episode!=row['episode']:
                episode=row['episode'];previous=None;progress=Progress(1,'cpu',profile)
                state=trace_state(row['pre_action_state'],None,'cpu');ref=MotorReference(state)
                progress.reset(torch.tensor([0]),state)
            state=trace_state(row['pre_action_state'],previous,'cpu')
            obs=observe(state,ref,progress)[0].numpy().copy()
            p=torch.tensor([row['commanded_positions_w_m']],dtype=torch.float32)
            q=canonical(torch.tensor([row['commanded_quaternions_wxyz']],dtype=torch.float32))
            jaw=torch.tensor([row['jaw_targets_rad']],dtype=torch.float32)
            action=torch.cat(((p-ref.p)/(.025*DT),rotvec(multiply(q,conjugate(ref.q)))/DT,
                              ((jaw-ref.jaw)/(.15*DT))[...,None]),-1).reshape(1,14)
            valid=(action.abs()<=1.001).float()[0].numpy()
            action=action.clamp(-1,1);ref.p=p;ref.q=q;ref.jaw=jaw
            post=trace_state([row['psm1'],row['psm2']],row['pre_action_state'],'cpu')
            before=progress.flags.clone();reward,success,failure,_=progress.update(post,action,'full')
            nxt=observe(post,ref,progress)[0].numpy().copy()
            lift_done=bool(progress.flags[0,1] and not before[0,1]);hand_done=bool(progress.flags[0,3] and not before[0,3])
            lift_reward=float(reward[0]-progress.gamma*progress.potential[0]) if lift_done else float(reward[0])
            vals=dict(obs=obs,actions=action[0].numpy(),next_obs=nxt,rewards=float(reward[0]),lift_rewards=lift_reward,done=bool(success[0]|failure[0]),
                episode=episode,bc_valid=valid,unassisted=row['peg_support']['holder']==0,
                lift_done=lift_done,handover_done=hand_done)
            for key,value in vals.items():arrays[key].append(value)
            previous=row['pre_action_state'];count+=1
            if count%3000==0:print(f'[SAC DATA] {count} transitions',flush=True)
    data={k:np.asarray(v,dtype=np.float32 if k in ('obs','actions','next_obs','rewards','lift_rewards','bc_valid') else None) for k,v in arrays.items()}
    ep=np.unique(data['episode']);train=data['episode']!=ep[-1]
    metadata=dict(schema=SCHEMA,observation_dim=OBS_DIM,action_dim=ACTION_DIM,episodes=ep.tolist(),
        train_episodes=ep[:-1].tolist(),validation_episode=int(ep[-1]),transitions=count,
        teacher_report=str(report),teacher_report_sha256=digest(report),teacher_trace_sha256=digest(trace),
        assisted_teacher=True,unassisted_critic_scope='prefix before first measured lift success; no fixture',
        unrepresentable_action_fraction=float(1-data['bc_valid'].mean()),
        velocity_estimate='backward difference for pre-action observation; post-action difference only for next observation',
        lift_events=int(data['lift_done'].sum()),handover_events=int(data['handover_done'].sum()),
        mean=data['obs'][train].mean(0).tolist(),std=np.maximum(data['obs'][train].std(0),.1).tolist())
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as file:np.savez_compressed(file,**data,metadata=json.dumps(metadata))
    print(json.dumps(metadata|dict(mean='saved in dataset',std='saved in dataset'),indent=2),flush=True)
    return metadata

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path)
    export(**vars(p.parse_args()))
