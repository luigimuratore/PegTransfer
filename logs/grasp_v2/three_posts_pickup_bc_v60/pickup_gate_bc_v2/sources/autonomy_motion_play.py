"""Replay only the last physically verified policy and speed in the curriculum."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--curriculum',type=Path,default=Path('logs/grasp_v2/autonomy_motion_positions_v42'))
    p.add_argument('--video',action='store_true');p.add_argument('--randomized',action='store_true')
    args,rest=p.parse_known_args()
    state=json.loads((args.curriculum/'status.json').read_text())
    if 'last_verified_policy' not in state:
        p.error('Comparison still pending: use checkpoint8-play to replay the preserved reference')
    policy=Path(state['last_verified_policy']);digest=hashlib.sha256(policy.read_bytes()).hexdigest()
    if digest!=state['last_verified_policy_sha256']:raise ValueError('Verified policy changed')
    level=state['levels'][state['last_passed_level']] if args.randomized else dict(jitter_mm=0,yaw_deg=0)
    speed=state['selected_speed'];mode='video' if args.video else 'play'
    folder=Path('logs/grasp_v2')/f'autonomy_verified_motion_{mode}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    command=[sys.executable,'-u',str(ROOT/'tools/peg_grasp_v2/autonomy_replay.py'),'play','--policy',str(policy),
        '--output',str(folder),'--skill','full','--num_envs','1','--episodes','1','--episode_steps','10000',
        '--sources','L5','--targets','R2','--seed','123','--jitter_mm',str(level['jitter_mm']),'--yaw_deg',str(level['yaw_deg']),
        '--grasp_support','1','--support_profile','firm','--placement_support',
        '--free_speed_scale',str(speed['free']),'--contact_speed_scale',str(speed['contact'])]
    command+=['--headless','--video'] if args.video else ['--realtime']
    if any(flag in rest for flag in ('--policy','--grasp_support','--free_speed_scale','--contact_speed_scale','--support_profile')):
        p.error('Use autonomy-play for custom experiments; this command restores verified settings')
    command+=rest
    print('Replaying last verified assisted BC policy: '+str(folder),flush=True)
    os.execv(sys.executable,command)

if __name__=='__main__':main()
