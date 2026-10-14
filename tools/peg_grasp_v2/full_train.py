"""CUDA Behavior Cloning of full-sequence motor gains from measured episodes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from full_learning import SCHEMA,OBS_DIM,ACTION_DIM,STAGES,load_data,decode_tensor,physics_hashes

def train(demonstrations,output,updates=6000,batch_size=1024,learning_rate=1e-4,seed=42,device='cuda:0'):
    import torch
    from torch_backend import resolve_device
    if output.exists():raise ValueError('Output exists; choose a new model directory')
    if updates<1 or batch_size<1 or not 0<learning_rate<.1:raise ValueError('Invalid training parameters')
    data,meta=load_data(demonstrations)
    from replay_full import verify
    _,bundle=verify(Path(meta['base_manifest']))
    donor_checkpoint=bundle/'donor_checkpoint/policy.npz'
    if physics_hashes(Path(__file__).parent)!=meta['physics_hashes']:
        raise ValueError('Physical sources differ from verified teacher')
    target=resolve_device(device);torch.set_num_threads(4);torch.manual_seed(seed)
    print(f'[FULL BC] Device {target}; '+(torch.cuda.get_device_name(target) if target.type=='cuda' else 'explicit CPU'),flush=True)
    episodes=sorted(meta['episodes']);validation=data['episode']==episodes[-1];training=~validation
    mean=data['observations'][training].mean(0).astype(np.float32)
    std=np.maximum(data['observations'][training].std(0),.02).astype(np.float32)
    raw=torch.as_tensor(data['observations'],device=target)
    x=(raw-torch.as_tensor(mean,device=target))/torch.as_tensor(std,device=target)
    y=torch.as_tensor(data['actions'],device=target)
    allowed=torch.as_tensor(data['motion_allowed'],device=target)
    net=torch.nn.Sequential(torch.nn.Linear(OBS_DIM,256),torch.nn.ReLU(),torch.nn.Linear(256,256),
        torch.nn.ReLU(),torch.nn.Linear(256,ACTION_DIM),torch.nn.Tanh()).to(target)
    optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate)
    groups=[np.flatnonzero(training&(data['stage']==stage)) for stage in range(len(STAGES))]
    lengths=torch.as_tensor([len(g) for g in groups],device=target)
    table=np.zeros((len(STAGES),max(map(len,groups))),dtype=np.int64)
    for stage,group in enumerate(groups):table[stage,:len(group)]=group
    table=torch.as_tensor(table,device=target)
    def prediction(indices):return decode_tensor(raw[indices],net(x[indices]),meta['parameters'])
    def objective(indices):
        mask=torch.ones_like(y[indices]);mask[~allowed[indices],:3]=0.
        return ((prediction(indices)-y[indices]).square()*mask).sum()/mask.sum()
    history=[]
    for update in range(1,updates+1):
        stages=torch.randint(len(STAGES),(batch_size,),device=target)
        offsets=(torch.rand(batch_size,device=target)*lengths[stages]).long()
        indices=table[stages,offsets]
        loss=objective(indices)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        optimizer.step()
        if update%500==0 or update==updates:
            history.append(dict(update=update,train_mse=float(loss.detach())))
            print(f'[FULL BC] {update}/{updates}; loss={float(loss.detach()):.6f}',flush=True)
    with torch.inference_mode():
        indices=torch.as_tensor(np.flatnonzero(validation),device=target)
        pred=prediction(indices).cpu().numpy();expected=data['actions'][validation]
    squared=(pred-expected)**2
    mask=np.ones_like(squared);mask[~data['motion_allowed'][validation],:3]=0
    errors=squared*mask
    by_stage={name:float(errors[data['stage'][validation]==stage].sum()/mask[data['stage'][validation]==stage].sum())
        for stage,name in enumerate(STAGES)}
    manifest=dict(schema=SCHEMA,algorithm='Behavior Cloning of positive XYZ and jaw feedback gains',
        parameters=meta['parameters'],physics_hashes=meta['physics_hashes'],base_manifest=meta['base_manifest'],
        base_manifest_sha256=meta['base_manifest_sha256'],demonstration_path=str(demonstrations.resolve()),
        demonstration_sha256=hashlib.sha256(demonstrations.read_bytes()).hexdigest(),
        demonstration_report_sha256=meta['report_sha256'],training_episodes=episodes[:-1],validation_episode=episodes[-1],
        settings=dict(updates=updates,batch_size=batch_size,learning_rate=learning_rate,seed=seed,net_arch=[256,256],
            observation_dim=OBS_DIM,action_dim=ACTION_DIM,device=str(target),gpu_name=torch.cuda.get_device_name(target) if target.type=='cuda' else None,
            sampling='equal phases, complete episode held out'),
        learned_control='positive XYZ gains (5..100/s) and both jaw-feedback gains (.5..1.5), all 20 phases',
        programmed_control='stage order/goals, grasp selection, orientation/IK, inactive-arm position, contact guards and fixed fixture',
        assisted_grasp=True,physical_evaluation_required=True,end_to_end_policy=False,
        donor_component='donor_policy.npz',donor_component_sha256=hashlib.sha256(donor_checkpoint.read_bytes()).hexdigest(),
        architecture_scope='frozen validated donor pickup actor plus trained full-sequence actor for transfer/receiver/placement',
        validation=dict(normalized_action_mse=float(errors.sum()/mask.sum()),by_stage_mse=by_stage,
            scope='one held-out episode in fixed L5 to R2 scene; no domain randomization'),history=history)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(donor_checkpoint,output/'donor_policy.npz')
    arrays=dict(mean=mean,std=std,metadata=json.dumps(manifest))
    for i,layer in enumerate((net[0],net[2],net[4])):
        arrays[f'w{i}']=layer.weight.detach().cpu().numpy();arrays[f'b{i}']=layer.bias.detach().cpu().numpy()
    with (output/'policy.npz').open('xb') as file:np.savez_compressed(file,**arrays)
    (output/'sources').mkdir()
    for source in sorted(Path(__file__).parent.glob('*.py'))+sorted(Path(__file__).parent.glob('*.json')):
        shutil.copy2(source,output/'sources'/source.name)
    manifest['files_sha256']={str(path.relative_to(output)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob('*')) if path.is_file()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'[FULL BC] Saved {output}/policy.npz; validation MSE={manifest["validation"]["normalized_action_mse"]:.6f}; Isaac evaluation required',flush=True)
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demonstrations',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=6000);p.add_argument('--batch_size',type=int,default=1024)
    p.add_argument('--learning_rate',type=float,default=1e-4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda:0')
    train(**vars(p.parse_args()))

if __name__=='__main__':main()
