"""Fine-tune only learned donor approach poses from explicitly assisted data.

The full router, jaw policies, grasp/lift/transfer heads and embedded pickup
decisions remain byte-identical. Whole successful episodes are split before
fitting. Physical validation is mandatory; offline loss cannot promote a model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from autonomy_learning import load_policy, SCHEMA
from autonomy_main import snapshot

ROOT=Path(__file__).resolve().parents[2]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def approach_samples(path, device):
    with np.load(path,allow_pickle=False) as archive:
        meta=json.loads(str(archive['metadata']))
        if (meta.get('schema')!=SCHEMA or meta.get('skill')!='full'
            or meta.get('collection_complete') is not True or meta.get('assisted_grasp') is not True):
            raise ValueError('Explicitly assisted, complete full-task demonstrations required')
        data={k:torch.as_tensor(archive[k],device=device) for k in ('obs','stage','episode','validation')}
    report_path=path.parent/'evaluation.json'
    report=json.loads(report_path.read_text());close=json.loads((path.parent/'environment_close.json').read_text())
    if (report.get('complete') is not True or report.get('assisted_grasp') is not True
        or close.get('environment_closed') is not True or close.get('evaluation_sha256')!=sha(report_path)):
        raise ValueError('Physical collection and verified clean closure required')
    if len(meta.get('episodes',[]))<8:raise ValueError('At least eight completed successful episodes required')
    successful={i for i,row in enumerate(report['cases']) if row.get('success') and row.get('full_success')}
    if set(meta['episodes'])!=successful or set(data['episode'].cpu().tolist())!=successful:
        raise ValueError('Only complete successful physical episodes may train')
    if data['obs'].ndim!=2 or data['obs'].shape[1]!=98 or not bool(torch.isfinite(data['obs']).all()):
        raise ValueError('Invalid physical observations')
    train_episodes=set(data['episode'][~data['validation'].bool()].cpu().tolist())
    valid_episodes=set(data['episode'][data['validation'].bool()].cpu().tolist())
    if not train_episodes or not valid_episodes or train_episodes&valid_episodes:
        raise ValueError('Separate whole training/validation episodes required')
    x=[];options=[];goals=[];heldout=[]
    for ep in sorted(successful):
        ids=(data['episode']==ep).nonzero().flatten()
        stages=data['stage'][ids]
        starts=torch.cat((ids.new_zeros(1),(stages[1:]!=stages[:-1]).nonzero().flatten()+1))
        ends=torch.cat((starts[1:],starts.new_tensor([len(ids)])))
        for begin,end in zip(starts.tolist(),ends.tolist()):
            stage=int(stages[begin])
            if stage not in (0,1,2) or end==len(ids) or int(stages[end])!=stage+1:continue
            x.append(data['obs'][ids[begin]].clone());options.append(stage)
            # Measured arrival pose immediately before the teacher transitions.
            goals.append(data['obs'][ids[end-1],:7].clone());heldout.append(ep in valid_episodes)
    if not x:raise ValueError('No completed donor approach visits')
    return (torch.stack(x),torch.tensor(options,device=device),torch.stack(goals),
        torch.tensor(heldout,device=device),meta)

def fit(policy,dataset,output,device='cuda:0',updates=2000):
    if not device.startswith('cuda') or not torch.cuda.is_available():raise ValueError('CUDA required')
    torch.set_num_threads(8);torch.manual_seed(42)
    manifest=json.loads((policy.parent/'manifest.json').read_text())
    if sha(policy)!=manifest['policy_sha256']:raise ValueError('Parent weights changed')
    for name,digest in manifest['source_hashes'].items():
        if sha(policy.parent/'sources'/name)!=digest:raise ValueError('Parent sources changed')
    collection=json.loads((dataset.parent/'manifest.json').read_text())
    if collection['physics_manifest_sha256']!=manifest['physics_manifest_sha256']:raise ValueError('Mixed physics')
    x,option,target,valid,meta=approach_samples(dataset,device)
    net=load_policy(policy,device);before={k:v.detach().clone() for k,v in net.state_dict().items()}
    if net.pickup_decoder is None or not net.pickup_decoder.linear_goals:raise ValueError('Structured pickup decoder required')
    net.requires_grad_(False);head=net.pickup_decoder.pose_head
    head.requires_grad_(True)
    # Only donor XYZ/quaternion rows for the three learned approach options.
    mask=torch.zeros_like(head.weight);biasmask=torch.zeros_like(head.bias)
    for stage in (0,1,2):mask[stage*14:stage*14+7]=1;biasmask[stage*14:stage*14+7]=1
    opt=torch.optim.Adam(head.parameters(),lr=2e-5)
    train=(~valid).nonzero().flatten();held=valid.nonzero().flatten()
    if len(train)<3 or len(held)<3:raise ValueError('Insufficient whole held-out visits')
    weight=x.new_tensor([100.,100.,100.,10.,10.,10.,10.])
    history=[]
    for update in range(1,updates+1):
        _,heads=net.pickup_decoder.outputs(x[train])
        pred=heads[torch.arange(len(train),device=device),option[train],:7]
        loss=((pred-target[train]).square()*weight).mean()
        # Keep changes small around the previously working decoder.
        prefix='pickup_decoder.pose_head.'
        loss+=.01*((head.weight-before[prefix+'weight']).square().mean()+(head.bias-before[prefix+'bias']).square().mean())
        opt.zero_grad();loss.backward();head.weight.grad.mul_(mask);head.bias.grad.mul_(biasmask)
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
        if update%500==0 or update==updates:
            with torch.no_grad():
                _,h=net.pickup_decoder.outputs(x[held]);p=h[torch.arange(len(held),device=device),option[held],:7]
                rmse=float((p[:,:3]-target[held,:3]).square().mean().sqrt()*100)
            history.append(dict(update=update,loss=float(loss.detach()),heldout_xyz_rmse_mm=rmse))
            print('[POSITION BC] '+json.dumps(history[-1]),flush=True)
    after=net.state_dict();changed=[]
    for key,value in before.items():
        if not torch.equal(after[key],value):changed.append(key)
    if any(key not in ('pickup_decoder.pose_head.weight','pickup_decoder.pose_head.bias') for key in changed):
        raise ValueError('Protected neural parameters changed')
    if not torch.equal(head.weight[mask==0],before['pickup_decoder.pose_head.weight'][mask==0]):raise ValueError('Protected pose rows changed')
    if not torch.equal(head.bias[biasmask==0],before['pickup_decoder.pose_head.bias'][biasmask==0]):raise ValueError('Protected pose biases changed')
    output.mkdir(parents=True,exist_ok=False)
    saved=torch.load(policy,map_location=device,weights_only=True);saved['state_dict']=after;torch.save(saved,output/'policy.pt')
    training=dict(algorithm='Limited supervised BC fine-tuning of donor approach poses',assisted_training=True,
        dataset_sha256=sha(dataset),parent_policy_sha256=sha(policy),updates=updates,learning_rate=2e-5,
        training_visits=len(train),validation_visits=len(held),whole_episode_split=True,
        changed_tensors=changed,protected_motor_gate_jaw_and_transfer_parameters_unchanged=True,
        physical_evaluation_required=True,native_full_policy_verified=False,gpu=torch.cuda.get_device_name(),history=history)
    (output/'training.json').write_text(json.dumps(training,indent=2)+'\n')
    manifest.update(source_hashes=snapshot(output),policy_sha256=sha(output/'policy.pt'),parent_policy_sha256=sha(policy),
        assisted_training=True,assisted_grasp=True,retrained=True,native_full_policy_verified=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return training

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--device',default='cuda:0');args=p.parse_args()
    if args.updates<1:p.error('Positive update count required')
    fit(args.policy,args.dataset,args.output,args.device,args.updates)

if __name__=='__main__':main()
