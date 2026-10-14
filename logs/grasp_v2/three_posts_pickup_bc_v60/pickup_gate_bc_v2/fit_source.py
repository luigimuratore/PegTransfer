"""BC of donor close/lift decisions from measured multi-post lift episodes.

Only two rows of the donor neural gate change. Motor poses, jaws, other
pickup decisions and the complete receiver/placement actor stay untouched.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from autonomy_three_posts import sha,replace_once

ROOT=Path(__file__).resolve().parents[2]

def fit(policy,dataset,output,updates=3000):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(8);torch.manual_seed(60001);device='cuda:0'
    manifest=json.loads((policy.parent/'manifest.json').read_text())
    if sha(policy)!=manifest['policy_sha256']:raise ValueError('Parent weights changed')
    for name,digest in manifest['source_hashes'].items():
        if sha(policy.parent/'sources'/name)!=digest:raise ValueError('Parent sources changed')
    path=dataset.parent/'evaluation.json';r=json.loads(path.read_text());close=json.loads((dataset.parent/'environment_close.json').read_text())
    if (r.get('complete') is not True or r.get('skill')!='lift' or r.get('teacher_at_inference') is not True
        or close.get('environment_closed') is not True or close.get('evaluation_sha256')!=sha(path)):
        raise ValueError('Complete measured expert lift collection required')
    cases=r['cases'];successful={i for i,c in enumerate(cases) if c.get('success') and c.get('grasp_psm1') and c.get('lift_stable')}
    with np.load(dataset,allow_pickle=False) as f:
        meta=json.loads(str(f['metadata']))
        if meta.get('collection_complete') is not True or set(meta['episodes'])!=successful or len(successful)<8:
            raise ValueError('Eight complete physical lifts required')
        arrays=[{k:f[k].copy() for k in ('obs','stage','episode','validation')}]
    source_counts={}
    for ep in successful:
        ids=arrays[0]['episode']==ep
        if int(ids.sum())!=cases[ep]['length'] or len(np.unique(arrays[0]['validation'][ids]))!=1:
            raise ValueError('Incomplete trajectory or split leakage')
        source=cases[ep]['reset']['source'];source_counts[source]=source_counts.get(source,0)+1
    if set(source_counts)!=set(['L5','L6','L4']):raise ValueError('Every requested pickup post needs actual measured lifts')
    # Replay original whole TRAIN/VALID episodes to protect earlier decisions.
    offset=10000
    for name,digest in manifest['dataset_hashes'].items():
        p=ROOT/name
        if sha(p)!=digest:raise ValueError('Historical full demonstration changed')
        with np.load(p,allow_pickle=False) as f:
            m=json.loads(str(f['metadata']))
            if m.get('collection_complete') is not True:raise ValueError('Historical collection incomplete')
            item={k:f[k].copy() for k in ('obs','stage','episode','validation')}
        item['episode']+=offset;offset=int(item['episode'].max())+1
        mask=item['stage']<6;arrays.append({k:v[mask] for k,v in item.items()})
    data={k:torch.as_tensor(np.concatenate([a[k] for a in arrays]),device=device) for k in arrays[0]}
    if data['obs'].shape[1]!=98 or not bool(torch.isfinite(data['obs']).all()):raise ValueError('Bad physical inputs')
    if set(data['episode'][data['validation']].tolist())&set(data['episode'][~data['validation']].tolist()):raise ValueError('Whole episode split leakage')
    # Load precisely the archived learning implementation, not mutable code.
    import importlib.util,sys
    sys.path.insert(0,str(policy.parent/'sources'))
    spec=importlib.util.spec_from_file_location('pickup_gate_archived_learning',policy.parent/'sources/autonomy_learning.py')
    learning=importlib.util.module_from_spec(spec);spec.loader.exec_module(learning)
    net=learning.load_policy(policy,device);before={k:v.clone() for k,v in net.state_dict().items()}
    net.requires_grad_(False);gate=net.pickup_decoder.gate;gate.requires_grad_(True)
    with torch.inference_mode():
        decoder=net.pickup_decoder
        features=decoder.encode(((data['obs']-decoder.mean)/decoder.std).clamp(-10,10)*decoder.input_mask).clone()
        reference=gate(features).clone()
    # The detached features become regular training tensors.
    features=features.clone();reference=reference.clone()
    train=(~data['validation']).nonzero().flatten();valid=data['validation'].nonzero().flatten()
    if not len(train) or not len(valid):raise ValueError('Separate held-out episodes required')
    groups=[]
    for stage in range(6):
        for new in (False,True):
            ids=train[(data['stage'][train]==stage)&((data['episode'][train]<10000)==new)]
            if len(ids):groups.append(ids)
    weights=torch.zeros_like(gate.weight);weights[3:5]=1
    biases=torch.zeros_like(gate.bias);biases[3:5]=1
    optimizer=torch.optim.Adam(gate.parameters(),lr=1e-4);history=[]
    for step in range(1,updates+1):
        ids=torch.cat([g[torch.randint(len(g),(max(1,1024//len(groups)),),device=device)] for g in groups])
        scores=gate(features[ids]);loss=torch.nn.functional.cross_entropy(scores,data['stage'][ids].long())
        historical=data['episode'][ids]>=10000
        if bool(historical.any()):
            loss+=.02*torch.nn.functional.kl_div(scores[historical].log_softmax(-1),reference[ids][historical].softmax(-1),reduction='batchmean')
        optimizer.zero_grad();loss.backward();gate.weight.grad.mul_(weights);gate.bias.grad.mul_(biases)
        optimizer.step()
        if step%500==0 or step==updates:
            with torch.no_grad():
                accuracy=float((gate(features[valid]).argmax(-1)==data['stage'][valid]).float().mean())
            history.append(dict(update=step,loss=float(loss.detach()),heldout_accuracy=accuracy))
            print('[PICKUP GATE BC] '+json.dumps(history[-1]),flush=True)
    changed={k for k,v in before.items() if not torch.equal(v,net.state_dict()[k])}
    if changed!={'pickup_decoder.gate.weight','pickup_decoder.gate.bias'}:raise ValueError('Protected neural parameters changed')
    for key,mask in (('pickup_decoder.gate.weight',weights),('pickup_decoder.gate.bias',biases)):
        if not torch.equal(before[key][mask==0],net.state_dict()[key][mask==0]):raise ValueError('Protected decision rows changed')
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,output/'fit_source.py')
    # The new expert data describe the close/lift boundary. Keep the original
    # learned approach and post-lift gate, using the network's own prior choice.
    net.register_buffer('pickup_gate_original_weight',before['pickup_decoder.gate.weight'].clone())
    net.register_buffer('pickup_gate_original_bias',before['pickup_decoder.gate.bias'].clone())
    saved=torch.load(policy,map_location=device,weights_only=True);saved['state_dict']=net.state_dict();torch.save(saved,output/'policy.pt')
    shutil.copytree(policy.parent/'sources',output/'sources',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    source_path=output/'sources/autonomy_learning.py';source=source_path.read_text()
    marker='            pickup_logits,_=self.pickup_decoder.outputs(obs)'
    source=replace_once(source,marker,marker+"\n            if hasattr(self,'pickup_gate_original_weight'):\n                d=self.pickup_decoder\n                features=d.encode(((obs-d.mean)/d.std).clamp(-10,10)*d.input_mask)\n                original=torch.nn.functional.linear(features,self.pickup_gate_original_weight,self.pickup_gate_original_bias)\n                pickup_logits=torch.where((previous==3)[:,None],pickup_logits,original)")
    marker="    net.load_state_dict(state['state_dict']);return net.eval()"
    source=replace_once(source,marker,"    if 'pickup_gate_original_weight' in saved:\n        net.register_buffer('pickup_gate_original_weight',saved['pickup_gate_original_weight'].clone())\n        net.register_buffer('pickup_gate_original_bias',saved['pickup_gate_original_bias'].clone())\n"+marker)
    source_path.write_text(source)
    spec=importlib.util.spec_from_file_location('pickup_gate_new_learning',source_path)
    patched=importlib.util.module_from_spec(spec);spec.loader.exec_module(patched)
    loaded=patched.load_policy(output/'policy.pt',device)
    for k,v in net.state_dict().items():
        if not torch.equal(v,loaded.state_dict()[k]):raise ValueError('Save/load changed actor')
    manifest['source_hashes']['autonomy_learning.py']=sha(source_path)
    manifest.update(policy_sha256=sha(output/'policy.pt'),parent_policy_sha256=sha(policy),retrained=True,
        pickup_gate_dataset_sha256=sha(dataset))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    evidence=dict(algorithm='Measured multi-post donor decision BC; not SAC',updates=updates,learning_rate=1e-4,
        batch_size=1024,changed_tensors=sorted(changed),changed_decisions=[3,4],protected_tensors=len(before)-2,
        original_motor_jaw_and_full_router_weights_unchanged=True,whole_episode_split=True,
        new_gate_context='Only when the network previously chose close (3); original neural gate for other prior choices',
        donor_sources_measured_lifts=source_counts,dataset_sha256=sha(dataset),physical_collection_report_sha256=sha(path),
        dataset=str(dataset.resolve()),fit_source_sha256=sha(output/'fit_source.py'),
        gpu=torch.cuda.get_device_name(),history=history,physical_full_evaluation_required=True,
        training_episodes=data['episode'][train].unique().tolist(),validation_episodes=data['episode'][valid].unique().tolist(),
        assisted_training=True,native_full_policy_verified=False)
    (output/'training.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('Saved CUDA candidate',output/'policy.pt',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();fit(a.policy,a.dataset,a.output)

if __name__=='__main__':main()
