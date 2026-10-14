"""Replay only a physically accepted faster transit profile, with frozen code."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from autonomy_motion_curriculum import sha
from autonomy_transit_curriculum import verified_report

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,default=ROOT/'logs/grasp_v2/autonomy_transit_speed_v50')
    p.add_argument('--episodes',type=int,default=3);p.add_argument('--video',action='store_true')
    args=p.parse_args()
    if args.episodes<1 or (args.video and args.episodes!=1):p.error('Positive episodes; video requires --episodes1')
    status_path=args.experiment/'status.json'
    if not status_path.is_file():
        print(f'Nessuna prova disponibile in {args.experiment}.\nPer il modello verificato: bash peg_grasp_v2.sh checkpoint9-play --episodes {args.episodes}',file=sys.stderr)
        return 2
    state=json.loads(status_path.read_text());selected=state.get('selected')
    if not selected:
        message='Le prove sono concluse: nessuna velocità maggiore ha conservato i successi del riferimento.' if state.get('complete') else f"Le prove sono ancora in corso (fase: {state.get('stage')}). Nessuna velocità maggiore è ancora verificata."
        print(f'{message}\nPer il modello verificato: bash peg_grasp_v2.sh checkpoint9-play --episodes {args.episodes}',file=sys.stderr)
        return 2
    policy=ROOT/selected['policy'];digest=sha(policy)
    if digest!=selected['policy_sha256']:raise ValueError('Selected weights changed')
    for row in selected['reports']:
        path=ROOT/row['report']
        if sha(path)!=row['report_sha256']:raise ValueError('Selected report changed')
        fixed=row['label'].endswith('_fixed');seed=123 if fixed else int(row['label'].rsplit('seed',1)[1])
        report,ok=verified_report(path,digest,seed,0. if fixed else .5,0. if fixed else 2.,3 if fixed else 16)
        speed=report['motion_reference']
        if (not ok or speed['profile']!=selected['motion_profile'] or speed['free_scale']!=selected['free_speed_scale']
            or speed['transport_scale']!=selected['transport_speed_scale']):raise ValueError('Unverified selected speed profile')
    out=ROOT/'logs/grasp_v2'/f"transit_play_{datetime.now():%Y%m%d_%H%M%S_%f}"
    return subprocess.call([sys.executable,'-u',str(HERE/'autonomy_replay.py'),'play','--policy',str(policy),
        '--output',str(out),'--skill','full','--num_envs','1','--episodes',str(args.episodes),'--episode_steps','10000',
        '--sources','L5','--targets','R2','--jitter_mm','.5','--yaw_deg','2','--seed','2468',
        '--grasp_support','1','--support_profile','firm','--placement_support',
        '--motion_profile',selected['motion_profile'],'--free_speed_scale',str(selected['free_speed_scale']),
        '--transport_speed_scale',str(selected['transport_speed_scale']),
        *(['--headless','--video'] if args.video else ['--realtime'])],cwd=ROOT)


if __name__=='__main__':raise SystemExit(main())
