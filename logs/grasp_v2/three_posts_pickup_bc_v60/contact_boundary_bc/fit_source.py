"""Fit a contact-invariant neural close/lift choice; preserve all prior weights."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
from autonomy_three_posts import sha,replace_once

FEATURES="""def pickup_boundary_features(obs):
    forces=obs[:,21:27].reshape(-1,2,3)
    magnitudes=forces.norm(dim=-1)
    opposition=-(forces[:,0]*forces[:,1]).sum(-1)/(magnitudes.prod(-1)+1e-6)
    return torch.cat((magnitudes.sort(-1).values,opposition[:,None],
        obs[:,19:21],obs[:,34:35],obs[:,90:91],obs[:,94:95]),-1)

"""

def fit(policy,dataset,output):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(8);torch.manual_seed(60002);device='cuda:0'
    manifest=json.loads((policy.parent/'manifest.json').read_text())
    assert sha(policy)==manifest['policy_sha256']
    for name,digest in manifest['source_hashes'].items():
        assert sha(policy.parent/'sources'/name)==digest
    rpath=dataset.parent/'evaluation.json';report=json.loads(rpath.read_text())
    closure=json.loads((dataset.parent/'environment_close.json').read_text())
    assert report['complete'] and report['success']==report['episodes']==12 and report['teacher_at_inference']
    assert closure['environment_closed'] and closure['evaluation_sha256']==sha(rpath)
    with np.load(dataset,allow_pickle=False) as f:
        meta=json.loads(str(f['metadata']));assert meta['collection_complete']
        obs=f['obs'].copy();stage=f['stage'].copy();episode=f['episode'].copy();validation=f['validation'].copy()
    assert set(episode)==set(range(12))
    for ep,case in enumerate(report['cases']):
        mask=episode==ep
        assert int(mask.sum())==case['length'] and len(np.unique(validation[mask]))==1
        assert case['success'] and case['lift_stable']
    assert not(set(episode[validation])&set(episode[~validation]))
    mask=(stage==3)|(stage==4)
    x=torch.as_tensor(obs[mask],device=device);y=torch.as_tensor(stage[mask]-3,device=device).long()
    valid=torch.as_tensor(validation[mask],device=device);train=~valid
    namespace={'torch':torch};exec(FEATURES,namespace);features=namespace['pickup_boundary_features'](x)
    head=torch.nn.Sequential(torch.nn.Linear(8,32),torch.nn.Tanh(),torch.nn.Linear(32,32),
        torch.nn.Tanh(),torch.nn.Linear(32,2)).to(device)
    optimizer=torch.optim.Adam(head.parameters(),lr=3e-4)
    groups=[(train&(y==c)).nonzero().flatten() for c in (0,1)]
    assert all(len(g) for g in groups) and bool(valid.any())
    history=[]
    for update in range(1,2001):
        ids=torch.cat([g[torch.randint(len(g),(512,),device=device)] for g in groups])
        scores=head(features[ids]);loss=torch.nn.functional.cross_entropy(scores,y[ids])
        optimizer.zero_grad();loss.backward();optimizer.step()
        if update%500==0:
            with torch.no_grad():
                predictions=head(features[valid]).argmax(-1)
                per_class=[float((predictions[y[valid]==c]==c).float().mean()) for c in (0,1)]
            row=dict(update=update,loss=float(loss.detach()),heldout_close_lift_accuracy=per_class)
            history.append(row);print('[CONTACT BC] '+json.dumps(row),flush=True)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(policy.parent/'sources',output/'sources',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    path=output/'sources/autonomy_learning.py';source=path.read_text()
    source=replace_once(source,'def load_policy(path,device):',FEATURES+'def load_policy(path,device):')
    marker='            pickup_logits,_=self.pickup_decoder.outputs(obs)'
    source=replace_once(source,marker,marker+"\n            if hasattr(self,'pickup_boundary'):\n                scores=self.pickup_boundary(pickup_boundary_features(obs))\n                boundary=torch.full_like(pickup_logits,-torch.inf)\n                boundary[:,3:5]=scores\n                pickup_logits=torch.where((previous==3)[:,None],boundary,pickup_logits)")
    marker="    net.load_state_dict(state['state_dict']);return net.eval()"
    source=replace_once(source,marker,"    if 'pickup_boundary.0.weight' in saved:\n        net.pickup_boundary=torch.nn.Sequential(torch.nn.Linear(8,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,2)).to(device)\n"+marker)
    path.write_text(source)
    sys.path.insert(0,str(output/'sources'))
    spec=importlib.util.spec_from_file_location('boundary_archived_learning',path)
    learning=importlib.util.module_from_spec(spec);spec.loader.exec_module(learning)
    net=learning.load_policy(policy,device);before={k:v.clone() for k,v in net.state_dict().items()}
    net.pickup_boundary=head.eval()
    saved=torch.load(policy,map_location=device,weights_only=True);saved['state_dict']=net.state_dict()
    torch.save(saved,output/'policy.pt');loaded=learning.load_policy(output/'policy.pt',device)
    for name,value in before.items():assert torch.equal(value,loaded.state_dict()[name])
    for name,value in net.state_dict().items():assert torch.equal(value,loaded.state_dict()[name])
    # Outside the network's own close choice, the original decisions/motors
    # must be identical on full physical observations from every route stage.
    original=learning.load_policy(policy,device)
    with torch.inference_mode():
        test=torch.as_tensor(obs[::37],device=device)
        for prior in (0,1,2,4,5,6,9,12,16,18):
            original.reset();loaded.reset()
            original.selected_option=torch.full((len(test),),prior,device=device)
            loaded.selected_option=original.selected_option.clone()
            torch.testing.assert_close(original(test),loaded(test),atol=0,rtol=0)
    manifest['policy_sha256']=sha(output/'policy.pt');manifest['parent_policy_sha256']=sha(policy)
    manifest['source_hashes']['autonomy_learning.py']=sha(path)
    manifest['pickup_boundary_dataset_sha256']=sha(dataset);manifest['retrained']=True
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(__file__,output/'fit_source.py')
    evidence=dict(algorithm='BC contact-invariant donor close/lift choice; not SAC',updates=2000,
        learning_rate=3e-4,batch_size=1024,architecture=[8,32,32,2],gpu=torch.cuda.get_device_name(),
        original_motor_jaw_and_full_router_weights_unchanged=True,protected_tensors=len(before),
        features='Two sorted force magnitudes, opposition, two jaws, reference jaw, measured grasp and contact age',
        no_teacher_clock_or_source_post_input=True,new_gate_context='Only after the network previously selected close (3)',
        whole_episode_split=True,training_episodes=np.unique(episode[~validation]).tolist(),
        validation_episodes=np.unique(episode[validation]).tolist(),history=history,
        dataset=str(dataset.resolve()),dataset_sha256=sha(dataset),physical_collection_report_sha256=sha(rpath),
        physical_full_evaluation_required=True,native_full_policy_verified=False,assisted_training=True,
        fit_source_sha256=sha(output/'fit_source.py'))
    (output/'training.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('Saved neural contact-choice candidate',output/'policy.pt',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('policy','dataset','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();fit(a.policy,a.dataset,a.output)
if __name__=='__main__':main()
