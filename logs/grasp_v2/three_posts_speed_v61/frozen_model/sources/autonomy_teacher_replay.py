"""Replay a preserved native full teacher; explicitly not neural inference."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=REPO/'milestones/checkpoint6-native-full-teacher.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--headless',action='store_true');p.add_argument('--video',action='store_true')
    p.add_argument('--realtime',action='store_true')
    args=p.parse_args()
    if args.episodes<1 or (args.video and args.episodes!=1):p.error('Positive episodes; video requires one episode')
    record=json.loads(args.checkpoint.read_text());bundle=REPO/record['preserved_bundle']
    if record.get('teacher_at_inference') is not True or record.get('assisted_grasp') is not False:
        raise ValueError('Expected an explicitly native teacher checkpoint')
    for name,digest in record['files'].items():
        if hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Preserved file changed: {name}')
    manifest=json.loads((bundle/'manifest.json').read_text())
    physics=REPO/'milestones/checkpoint3-full-assisted.json'
    if hashlib.sha256(physics.read_bytes()).hexdigest()!=manifest['physics_manifest_sha256']:
        raise ValueError('Preserved physics manifest changed')
    folder=bundle/'sources';sys.path.insert(0,str(folder))
    baseline=importlib.import_module('replay_full');baseline.REPO=REPO
    module=importlib.import_module('autonomy_main');module.REPO=REPO;module.HERE=folder
    params=record['parameters']
    argv=['collect','--output',str(args.output),'--episodes',str(args.episodes),'--num_envs','1']
    for key in ('skill','episode_steps','seed','device','jitter_mm','yaw_deg','receiver_mode',
                'receiver_wall_angle','receiver_wall_depth','receiver_wall_tilt','receiver_open_angle',
                'receiver_release_bottom','receiver_pressure_min','receiver_pressure_max','donor_release_rate'):
        if params.get(key) is not None:argv.extend((f'--{key}',str(params[key])))
    for key in ('sources','targets'):argv.extend((f'--{key}',*params[key]))
    if params.get('fixed_donor_during_contact'):argv.append('--fixed_donor_during_contact')
    for key in ('headless','video','realtime'):
        if getattr(args,key):argv.append(f'--{key}')
    print('[NATIVE TEACHER] Preserved programmed controller; no neural policy and no peg fixture',flush=True)
    sys.argv=[str(folder/'autonomy_main.py'),*argv]
    return module.main()

if __name__=='__main__':raise SystemExit(main())
