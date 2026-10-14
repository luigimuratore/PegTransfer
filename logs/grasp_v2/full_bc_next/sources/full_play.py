"""Replay a full learned motor policy using its frozen sources and verified scene."""
import argparse
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]

def verify_model(policy):
    from replay_full import verify
    manifest=json.loads((policy.parent/'manifest.json').read_text())
    if policy.name!='policy.npz':raise ValueError('Use the saved policy.npz')
    for name,digest in manifest['files_sha256'].items():
        path=policy.parent/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Full-model source or weights changed: {path}')
    base=Path(manifest['base_manifest'])
    if hashlib.sha256(base.read_bytes()).hexdigest()!=manifest['base_manifest_sha256']:
        raise ValueError('Preserved baseline manifest changed')
    _,bundle=verify(base)
    return manifest,bundle

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,default=REPO/'logs/grasp_v2/full_bc2/policy.npz')
    p.add_argument('--check',action='store_true');p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--device',default='cuda:0');p.add_argument('--headless',action='store_true')
    p.add_argument('--realtime',action='store_true');p.add_argument('--video',action='store_true')
    p.add_argument('--video_path',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args();policy=args.policy.resolve();manifest,bundle=verify_model(policy)
    if args.check:
        print(f'Full BC weights, sources and baseline verified: {policy}');return 0
    if args.episodes<1 or (args.video and args.episodes!=1):p.error('Positive episodes; video requires one episode')
    stamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    output=(args.output or REPO/f'logs/grasp_v2/full_bc_inference_{stamp}.json').resolve()
    argv=['--full_policy',str(policy),'--robot_usd',str(bundle/'data/Isaac/Healthcare/0.5.0/132c82d/Robots/dVRK/PSM/psm.usd'),
        '--episodes',str(args.episodes),'--device',args.device,'--output',str(output)]
    for key,value in manifest['parameters'].items():
        if isinstance(value,bool):
            if value:argv.append('--'+key)
        else:argv+=['--'+key]+[str(v) for v in (value if isinstance(value,list) else [value])]
    if args.headless or args.video:argv.append('--headless')
    if args.realtime:argv.append('--realtime')
    if args.video:argv+=['--video_path',str((args.video_path or REPO/f'logs/grasp_v2/videos/full_bc_front_{stamp}.mp4').resolve())]
    scripts=bundle/'workflows/robotic_surgery/scripts';ext=scripts/'simulation/exts'
    sys.path[:0]=[str(policy.parent/'sources'),str(scripts),str(ext/'robotic.surgery.tasks'),str(ext/'robotic.surgery.assets')]
    full=importlib.import_module('full_task');full.REPO=bundle
    sys.argv=[str(policy.parent/'sources/full_task.py'),*argv]
    return full.main()

if __name__=='__main__':raise SystemExit(main())
