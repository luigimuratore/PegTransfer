"""Explicitly assisted BC of receiver jaws; preserve every existing tensor."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from autonomy_learning import load_policy, SCHEMA
from autonomy_main import snapshot

ROOT=Path(__file__).resolve().parents[2]
RECEIVER_RECIPE=dict(receiver_wall_angle=300.,receiver_wall_depth=4.,receiver_wall_tilt=0.,
    receiver_open_angle=.25,receiver_release_bottom=.03,receiver_pressure_min=.36,receiver_pressure_max=.42,
    donor_release_rate=.02,fixed_donor_during_contact=True,recapture_receiver_after_transfer=True)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recovery_samples(dataset,device):
    report_path=dataset.parent/'evaluation.json'
    report=json.loads(report_path.read_text())
    close=json.loads((dataset.parent/'environment_close.json').read_text())
    if (report.get('complete') is not True or report.get('assisted_grasp') is not True
        or report.get('teacher_at_inference') is not True
        or close.get('environment_closed') is not True or close.get('evaluation_sha256')!=sha(report_path)):
        raise ValueError('Complete assisted expert recovery and clean-closure receipt required')
    with np.load(dataset,allow_pickle=False) as archive:
        meta=json.loads(str(archive['metadata']))
        if (meta.get('schema')!=SCHEMA or meta.get('skill')!='full'
            or meta.get('collection_complete') is not True or meta.get('assisted_grasp') is not True):
            raise ValueError('Explicit assisted, complete full demonstrations required')
        params=meta.get('parameters',{})
        if any(params.get(key)!=value for key,value in RECEIVER_RECIPE.items()):
            raise ValueError('Receiver geometry/pressure recipe differs from the verified parent grasp')
        if not (params.get('receiver_recovery_handoff') is True and params.get('sync_learner_reference') is True
                and params.get('receiver_pressure_min')==.36 and params.get('receiver_pressure_max')==.42):
            raise ValueError('Synchronized receiver recovery with verified teacher pressure recipe required')
        prefix=params.get('learner_prefix_steps',0)
        if not isinstance(prefix,int) or prefix<=0:raise ValueError('Continuous learner prefix required')
        raw={key:archive[key].copy() for key in ('obs','actions','stage','episode','validation')}
    successful={i for i,row in enumerate(report['cases']) if row.get('success') and row.get('full_success')}
    if len(successful)<8 or set(meta['episodes'])!=successful or set(np.unique(raw['episode']).tolist())!=successful:
        raise ValueError('At least eight complete successful physical recoveries required')
    if raw['obs'].shape[1:]!=(98,) or raw['actions'].shape[1:]!=(14,):raise ValueError('Incompatible observations/actions')
    if not np.isfinite(raw['obs']).all() or not np.isfinite(raw['actions']).all():raise ValueError('Nonfinite data')
    sizes={len(x) for x in raw.values()}
    if len(sizes)!=1:raise ValueError('Mismatched dataset arrays')
    train_eps=set();valid_eps=set();selected=np.zeros(len(raw['obs']),dtype=bool)
    for ep in successful:
        ids=np.flatnonzero(raw['episode']==ep)
        split=np.unique(raw['validation'][ids])
        if len(split)!=1:raise ValueError('Whole-episode TRAIN/VALID split required')
        (valid_eps if split[0] else train_eps).add(ep)
        if len(ids)!=report['cases'][ep]['length']:raise ValueError('Episode length differs from physical report')
        # Discard all shadow expert labels before the actual handoff.
        selected[ids[prefix:]]=True
    selected&=np.isin(raw['stage'],[10,11])
    if not train_eps or len(valid_eps)<2 or train_eps&valid_eps:raise ValueError('Separate whole held-out recoveries required')
    data={key:torch.as_tensor(value[selected],device=device) for key,value in raw.items()}
    data['obs']=data['obs'].float();data['actions']=data['actions'].float()
    data['stage']=data['stage'].long();data['validation']=data['validation'].bool()
    if not len(data['obs']):raise ValueError('No physical receiver closure/release samples')
    reference=data['obs'][:,69]*.5
    # Recorded one-step expert jaw reference, never a future state or a
    # synthetic force label. The same generic jaw speed cap is used at replay.
    data['rate']=(((data['actions'][:,13]+1)*.25-reference)/(.15*.01)).clamp(-1,1)
    return data,meta


def attach_receiver_head(net):
    if net.receiver_jaw_rate_head is not None:raise ValueError('This experiment expects an unmodified parent receiver')
    net.receiver_jaw_rate_head=copy.deepcopy(net.jaw_rate_head)
    mask=torch.zeros(net.options,device=net.mean.device,dtype=torch.bool);mask[10:12]=True
    net.register_buffer('receiver_jaw_options',mask)


def fit(policy,dataset,output,device='cuda:0',updates=2000,batch_size=1024):
    if not device.startswith('cuda') or not torch.cuda.is_available():raise ValueError('CUDA required')
    if min(updates,batch_size)<1:raise ValueError('Positive updates and batch size required')
    torch.set_num_threads(8);torch.manual_seed(42)
    manifest=json.loads((policy.parent/'manifest.json').read_text())
    if sha(policy)!=manifest['policy_sha256']:raise ValueError('Parent weights changed')
    for name,digest in manifest['source_hashes'].items():
        if sha(policy.parent/'sources'/name)!=digest:raise ValueError('Parent source changed')
    collection=json.loads((dataset.parent/'manifest.json').read_text())
    if collection['physics_manifest_sha256']!=manifest['physics_manifest_sha256']:raise ValueError('Mixed physics')
    if collection.get('policy_sha256')!=sha(policy):raise ValueError('Recoveries do not originate from the parent actor')
    data,meta=recovery_samples(dataset,device)
    net=load_policy(policy,device)
    if net.options!=19 or net.pickup_decoder is None or not net.jaw_rate_head.coupled:
        raise ValueError('Preserved full actor with coupled categorical jaws required')
    # Check that this recipe actually matches the parent's fitted reach pose,
    # rather than accepting a different grasp just because pressures match.
    from geometry import contact_candidate
    from rl_core import rotvec,multiply,conjugate,canonical
    profile=json.loads((ROOT/'tools/peg_grasp_v2/peg_profile.json').read_text())
    candidate=contact_candidate(profile,'wall',0,.5,2,wall_angle_deg=300.,wall_depth_mm=4.,wall_tilt_deg=0.,wall_open_angle=.25)
    expected=torch.tensor(candidate['tip_local_m'],device=device)
    expectedq=torch.tensor(candidate['tool_quaternion_local_wxyz'],device=device)
    pose=net.relative_pose_goals[9,1]
    if (float((pose[:3]-expected).norm())>.0005
        or float(rotvec(multiply(canonical(pose[3:]),conjugate(expectedq))).norm())>.05):
        raise ValueError('Parent learned receiver reach differs from the recovery grasp recipe')
    before={key:value.clone() for key,value in net.state_dict().items()}
    net.requires_grad_(False);attach_receiver_head(net)
    net.receiver_jaw_rate_head.requires_grad_(True)
    labels=(data['rate'][:,None]-net.receiver_jaw_rate_head.rates).abs().argmin(-1)
    train=(~data['validation']).nonzero().flatten();valid=data['validation'].nonzero().flatten()
    if min(len(train),len(valid))<20:raise ValueError('Insufficient receiver observations')
    groups=[]
    for option in (10,11):
        for label in labels[train].unique():
            group=train[(data['stage'][train]==option)&(labels[train]==label)]
            if len(group):groups.append(group)
    optimizer=torch.optim.Adam(net.receiver_jaw_rate_head.parameters(),lr=5e-5)
    history=[]
    def objective(ids):
        prediction=net.receiver_jaw_rate_head(data['obs'][ids],data['stage'][ids],logits=True)[:,1]
        return nn.functional.cross_entropy(prediction,labels[ids])
    for update in range(1,updates+1):
        ids=torch.cat([g[torch.randint(len(g),(max(1,batch_size//len(groups)),),device=device)] for g in groups])
        loss=objective(ids)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.receiver_jaw_rate_head.parameters(),1.);optimizer.step()
        if update%500==0 or update==updates:
            with torch.no_grad():
                logits=net.receiver_jaw_rate_head(data['obs'][valid],data['stage'][valid],logits=True)[:,1]
                accuracy=float((logits.argmax(-1)==labels[valid]).float().mean())
                val_loss=float(nn.functional.cross_entropy(logits,labels[valid]))
            history.append(dict(update=update,loss=float(loss.detach()),validation_loss=val_loss,validation_accuracy=accuracy))
            print(f'[RECEIVER BC] {update}/{updates} loss={float(loss.detach()):.6f} heldout_accuracy={accuracy:.3f}',flush=True)
    after=net.state_dict()
    if any(not torch.equal(after[key],value) for key,value in before.items()):raise ValueError('Existing actor tensor changed')
    output.mkdir(parents=True,exist_ok=False)
    state=torch.load(policy,map_location=device,weights_only=True);state['state_dict']=after;torch.save(state,output/'policy.pt')
    check=load_policy(output/'policy.pt',device)
    with torch.inference_mode():
        torch.testing.assert_close(check.jaw_rates(data['obs'][valid],data['stage'][valid]),net.jaw_rates(data['obs'][valid],data['stage'][valid]),rtol=0,atol=0)
    config=dict(algorithm='Receiver-only supervised Behavior Cloning; not SAC',assisted_training=True,
        parent_policy_sha256=sha(policy),dataset_sha256=sha(dataset),updates=updates,batch_size=batch_size,learning_rate=5e-5,
        receiver_options=[10,11],recorded_expert_labels_only=True,pre_handoff_labels_excluded=True,
        original_tensors_unchanged=len(before),donor_and_motor_router_pose_parameters_unchanged=True,
        training_episodes=data['episode'][train].unique().tolist(),validation_episodes=data['episode'][valid].unique().tolist(),
        whole_episode_split=True,physical_evaluation_required=True,native_full_policy_verified=False,
        collection_seed=meta['parameters']['seed'],fresh_validation_seeds=[2468,54321],gpu=torch.cuda.get_device_name(),history=history)
    (output/'training.json').write_text(json.dumps(config,indent=2)+'\n')
    manifest.update(source_hashes=snapshot(output),policy_sha256=sha(output/'policy.pt'),parent_policy_sha256=sha(policy),
        dataset_sha256=sha(dataset),assisted_training=True,assisted_grasp=True,retrained=True,native_full_policy_verified=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,required=True);parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--updates',type=int,default=2000)
    args=parser.parse_args();fit(args.policy,args.dataset,args.output,updates=args.updates)


if __name__=='__main__':main()
