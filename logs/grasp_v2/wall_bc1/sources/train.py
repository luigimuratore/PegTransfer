"""Offline BC of wall translation/jaw velocity; never launches Isaac Sim."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from learning import SCHEMA,STAGES,OBS_DIM,ACTION_DIM,load_demonstrations


def train(demonstrations,output,updates=4000,batch_size=256,learning_rate=1e-4,seed=42):
    import torch
    if updates<1 or batch_size<1 or not np.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError('Positive finite training settings required')
    data,meta=load_demonstrations(demonstrations)
    episodes=sorted(meta['episodes'])
    if len(episodes)<3:
        raise ValueError('Collect at least 3 complete wall successes; reserve one episode for validation')
    output=Path(output)
    if output.exists():
        raise ValueError('Model output already exists; use a new directory')
    torch.set_num_threads(4);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    validation=data['episode']==episodes[-1];training=~validation
    mean=data['observations'][training].mean(0);std=np.maximum(data['observations'][training].std(0),.02)
    observations=torch.from_numpy((data['observations']-mean)/std)
    actions=torch.from_numpy(data['actions'])
    net=torch.nn.Sequential(torch.nn.Linear(OBS_DIM,128),torch.nn.ReLU(),
        torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,ACTION_DIM),torch.nn.Tanh())
    optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate)
    groups=[np.flatnonzero(training & (data['stage']==stage)) for stage in range(len(STAGES))]
    history=[]
    for update in range(1,updates+1):
        # Each phase gets equal sampling; long lift/hold phases must not drown
        # out the brief close/release decisions.
        sampled=rng.integers(len(STAGES),size=batch_size)
        indices=np.array([rng.choice(groups[stage]) for stage in sampled])
        prediction=net(observations[indices]);loss=(prediction-actions[indices]).square().mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        optimizer.step()
        if update%500==0 or update==updates:
            history.append(dict(update=update,train_mse=float(loss.detach())))
            print(f'[WALL BC] {update}/{updates}; loss={float(loss.detach()):.6f}',flush=True)
    with torch.inference_mode():
        prediction=net(observations[validation]).numpy();expected=data['actions'][validation]
    errors=(prediction-expected)**2
    per_stage={name:float(errors[data['stage'][validation]==i].mean()) for i,name in enumerate(STAGES)}
    manifest=dict(schema=SCHEMA,algorithm='Behavior Cloning',experiment=meta['experiment'],
        demonstration_path=str(Path(demonstrations).resolve()),demonstration_sha256=hashlib.sha256(Path(demonstrations).read_bytes()).hexdigest(),
        training_episodes=episodes[:-1],validation_episode=episodes[-1],
        settings=dict(updates=updates,batch_size=batch_size,learning_rate=learning_rate,seed=seed,net_arch=[128,128],sampling='balanced by phase'),
        learned_control='active-arm xyz reference velocity and coupled jaw velocity',
        programmed_control='grasp choice, phase order, orientation/IK, inactive arm, contact/slip guards',
        physical_evaluation_required=True,neural_physical_success_verified=False,
        validation=dict(normalized_action_mse=float(errors.mean()),by_stage_mse=per_stage,
                        scope='held-out episode, same fixed L5/PSM/pose; no domain randomization validation'),history=history)
    output.mkdir(parents=True,exist_ok=False)
    arrays=dict(mean=mean,std=std,metadata=json.dumps(manifest))
    for i,layer in enumerate((net[0],net[2],net[4])):
        arrays[f'w{i}']=layer.weight.detach().numpy();arrays[f'b{i}']=layer.bias.detach().numpy()
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
    p.add_argument('--batch_size',type=int,default=256)
    p.add_argument('--learning_rate',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    train(**vars(args))


if __name__=='__main__':main()
