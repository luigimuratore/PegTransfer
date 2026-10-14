"""Replay a SAC checkpoint with its exact saved sources."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import hashlib

REPO=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('mode',choices=['play','evaluate']);p.add_argument('--checkpoint',type=Path,required=True)
    args,rest=p.parse_known_args();checkpoint=args.checkpoint.resolve()
    manifest=json.loads((checkpoint.parent/'manifest.json').read_text());folder=checkpoint.parent/'sources'
    for name,digest in manifest['source_hashes'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Changed frozen source: {name}')
    if checkpoint.name=='final.zip' and hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint weights changed')
    sys.path.insert(0,str(folder));module=importlib.import_module('rl_main');module.REPO=REPO;module.HERE=folder
    sys.argv=[str(folder/'rl_main.py'),args.mode,'--checkpoint',str(checkpoint),*rest]
    return module.main()

if __name__=='__main__':raise SystemExit(main())
