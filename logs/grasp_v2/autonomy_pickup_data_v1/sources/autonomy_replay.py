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
    args,rest=p.parse_known_args();policy=args.policy.resolve();folder=policy.parent/'sources'
    manifest=json.loads((policy.parent/'manifest.json').read_text())
    if hashlib.sha256(policy.read_bytes()).hexdigest()!=manifest['policy_sha256']:raise ValueError('Weights changed')
    for name,digest in manifest['source_hashes'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Frozen source changed: {name}')
    sys.path.insert(0,str(folder))
    baseline=importlib.import_module('replay_full');baseline.REPO=REPO
    module=importlib.import_module('autonomy_main');module.REPO=REPO;module.HERE=folder
    sys.argv=[str(folder/'autonomy_main.py'),args.mode,'--policy',str(policy),*rest]
    return module.main()

if __name__=='__main__':raise SystemExit(main())
