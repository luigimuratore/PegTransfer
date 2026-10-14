"""Run a BC model using only its hash-verified preserved sources."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('mode',choices=('evaluate','play'));p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--diagnostic_trace',type=Path,help='Optional observation logger; does not alter actor inputs/actions')
    args,rest=p.parse_known_args();policy=args.policy.resolve();folder=policy.parent/'sources'
    manifest=json.loads((policy.parent/'manifest.json').read_text())
    if hashlib.sha256(policy.read_bytes()).hexdigest()!=manifest['policy_sha256']:raise ValueError('Weights changed')
    if hashlib.sha256((REPO/'milestones/checkpoint3-full-assisted.json').read_bytes()).hexdigest()!=manifest['physics_manifest_sha256']:
        raise ValueError('Preserved physics manifest changed')
    for name,digest in manifest['source_hashes'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Frozen source changed: {name}')
    sys.path.insert(0,str(folder))
    baseline=importlib.import_module('replay_full');baseline.REPO=REPO
    module=importlib.import_module('autonomy_main');module.REPO=REPO;module.HERE=folder
    if args.diagnostic_trace:
        learning=importlib.import_module('autonomy_learning');original=learning.physical_observation
        trace=[];calls=[0]
        def logged_observation(obs):
            x=original(obs)
            if calls[0]%50==0:
                trace.append(dict(step=calls[0],observation=x[0].cpu().tolist()))
                args.diagnostic_trace.write_text(json.dumps(dict(diagnostic_only=True,policy=str(policy),samples=trace),indent=2)+'\n')
            calls[0]+=1
            return x
        learning.physical_observation=logged_observation
    sys.argv=[str(folder/'autonomy_main.py'),args.mode,'--policy',str(policy),*rest]
    return module.main()

if __name__=='__main__':raise SystemExit(main())
