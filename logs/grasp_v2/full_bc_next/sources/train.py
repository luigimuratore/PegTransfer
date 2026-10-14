"""CUDA BC of positive wall position/pressure gains; no Isaac Sim."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from learning import SCHEMA,STAGES,OBS_DIM,ACTION_DIM,load_demonstrations,dataset_experiment,stage_speeds,MotorPolicy


def train(demonstrations,output,updates=4000,batch_size=1024,learning_rate=1e-4,seed=42,initialize_from=None,device='auto'):
    import torch
    from torch_backend import resolve_device,decode_action
    if updates<1 or batch_size<1 or not np.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError('Positive finite training settings required')
    data,meta=load_demonstrations(demonstrations)
    experiment=dataset_experiment(meta,initialize_from)
    episodes=sorted(meta['episodes'])
    if len(episodes)<3:
        raise ValueError('Collect at least 3 complete wall successes; reserve one episode for validation')
    output=Path(output)
    if output.exists():
        raise ValueError('Model output already exists; use a new directory')
    target_device=resolve_device(device)
    torch.set_num_threads(4);torch.manual_seed(seed)
    print(f'[WALL BC] Device: {target_device}'+(f'; {torch.cuda.get_device_name(target_device)}' if target_device.type=='cuda' else ''),flush=True)
    validation=data['episode']==episodes[-1];training=~validation
    mean=data['observations'][training].mean(0);std=np.maximum(data['observations'][training].std(0),.02)
    observations=torch.as_tensor((data['observations']-mean)/std,device=target_device)
    actions=torch.as_tensor(data['actions'],device=target_device)
    net=torch.nn.Sequential(torch.nn.Linear(OBS_DIM,128),torch.nn.ReLU(),
        torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,ACTION_DIM),torch.nn.Tanh()).to(target_device)
    if initialize_from:
        prior=MotorPolicy(initialize_from)
        # Normalization remains tied to this training split; reuse hidden
        # features only if exactly the same. Reuse matching position gains;
        # reset a legacy jaw-velocity head because it now predicts a gain.
        if np.array_equal(prior.mean,mean) and np.array_equal(prior.std,std):
            with torch.no_grad():
                for layer,(weight,bias) in zip((net[0],net[2]),prior.layers[:2]):
                    layer.weight.copy_(torch.as_tensor(weight,device=target_device));layer.bias.copy_(torch.as_tensor(bias,device=target_device))
                from learning import POSITION_SCHEMA
                count=4 if prior.metadata['schema']==SCHEMA else 3 if prior.metadata['schema']==POSITION_SCHEMA else 0
                net[4].weight[:count].copy_(torch.as_tensor(prior.layers[2][0][:count],device=target_device))
                net[4].bias[:count].copy_(torch.as_tensor(prior.layers[2][1][:count],device=target_device))
    params=experiment.get('parameters') or {}
    speeds=torch.as_tensor(stage_speeds(params),device=target_device)
    raw_observations=torch.as_tensor(data['observations'],device=target_device)
    reference_error=raw_observations[:,:3]*.05-raw_observations[:,6:9]*.003
    # The physical guard, not the actor, pauses lift for pressure recovery.
    # Legacy datasets contain these zero-velocity labels but no guard flag.
    guarded_pause=((actions[:,:3].norm(dim=1)<1e-6)&(reference_error.norm(dim=1)>2e-5)
                   &torch.as_tensor(np.isin(data['stage'],[5,6,7]),device=target_device))

    def decode(indices):
        latent=net(observations[indices])
        return decode_action(raw_observations[indices],latent,speeds,params.get('grip_force',.35),
                             experiment.get('candidate',{}).get('open_angle_rad',.15))

    def objective(indices):
        prediction=decode(indices)
        mask=torch.ones_like(prediction);mask[guarded_pause[indices],:3]=0.
        return ((prediction-actions[indices]).square()*mask).sum()/mask.sum()
    optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate)
    groups=[np.flatnonzero(training & (data['stage']==stage)) for stage in range(len(STAGES))]
    lengths=torch.as_tensor([len(group) for group in groups],device=target_device)
    padded=np.zeros((len(STAGES),max(len(group) for group in groups)),dtype=np.int64)
    for stage,group in enumerate(groups):padded[stage,:len(group)]=group
    sampling_table=torch.as_tensor(padded,device=target_device)
    history=[]
    for update in range(1,updates+1):
        # Each phase gets equal sampling; long lift/hold phases must not drown
        # out the brief close/release decisions.
        sampled=torch.randint(len(STAGES),(batch_size,),device=target_device)
        offsets=(torch.rand(batch_size,device=target_device)*lengths[sampled]).long()
        indices=sampling_table[sampled,offsets]
        loss=objective(indices)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        optimizer.step()
        if update%500==0 or update==updates:
            history.append(dict(update=update,train_mse=float(loss.detach())))
            print(f'[WALL BC] {update}/{updates}; loss={float(loss.detach()):.6f}',flush=True)
    with torch.inference_mode():
        prediction=decode(torch.as_tensor(np.flatnonzero(validation),device=target_device)).cpu().numpy();expected=data['actions'][validation]
    errors=(prediction-expected)**2
    per_stage={name:float(errors[data['stage'][validation]==i].mean()) for i,name in enumerate(STAGES)}
    manifest=dict(schema=SCHEMA,algorithm='Behavior Cloning with positive position/pressure-feedback gains',experiment=experiment,
        demonstration_experiment=meta['experiment'],
        initialize_from=str(Path(initialize_from).resolve()) if initialize_from else None,
        demonstration_path=str(Path(demonstrations).resolve()),demonstration_sha256=hashlib.sha256(Path(demonstrations).read_bytes()).hexdigest(),
        training_episodes=episodes[:-1],validation_episode=episodes[-1],
        settings=dict(updates=updates,batch_size=batch_size,learning_rate=learning_rate,seed=seed,net_arch=[128,128],sampling='balanced by phase on training device',device=str(target_device),gpu_name=torch.cuda.get_device_name(target_device) if target_device.type=='cuda' else None),
        learned_control='positive xyz position gains (5..30/s) and positive jaw pressure/opening gain (0.5..1.5)',
        programmed_control='position and pressure feedback, grasp choice, phase order, orientation/IK, inactive arm, contact/slip guards',
        guarded_pause_translation_labels_ignored=int(guarded_pause.sum()),
        physical_evaluation_required=True,neural_physical_success_verified=False,
        validation=dict(normalized_action_mse=float(errors.mean()),by_stage_mse=per_stage,
                        scope='held-out episode, same fixed L5/PSM/pose; no domain randomization validation'),history=history)
    output.mkdir(parents=True,exist_ok=False)
    arrays=dict(mean=mean,std=std,metadata=json.dumps(manifest))
    for i,layer in enumerate((net[0],net[2],net[4])):
        arrays[f'w{i}']=layer.weight.detach().cpu().numpy();arrays[f'b{i}']=layer.bias.detach().cpu().numpy()
    with (output/'policy.npz').open('xb') as file:np.savez_compressed(file,**arrays)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for source in sorted(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.json')):
        destination=output/'sources'/source.name;destination.parent.mkdir(exist_ok=True);shutil.copy2(source,destination)
    print(f'[WALL BC] Saved: {output / "policy.npz"}; requires Isaac evaluation',flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demonstrations',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=4000)
    p.add_argument('--batch_size',type=int,default=1024)
    p.add_argument('--learning_rate',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='auto',help='Default requires CUDA; --device cpu is an explicit fallback for code checks')
    p.add_argument('--initialize_from',type=Path,help='Prior wall model with verified physical source snapshot; allows reusing its demonstrations')
    args=p.parse_args()
    train(**vars(args))


if __name__=='__main__':main()
