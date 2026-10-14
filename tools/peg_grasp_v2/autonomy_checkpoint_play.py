"""Replay a preserved checkpoint using its archived settings and verified files."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
from preserve_autonomy import verify_archive

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint');p.add_argument('mode',choices=('play','video'))
    a,extra=p.parse_known_args()
    matches=list((ROOT/'milestones').glob(a.checkpoint+'-*.json'))
    if len(matches)!=1:raise ValueError('A unique preserved checkpoint is required')
    milestone=matches[0];m=verify_archive(milestone)
    output=ROOT/'logs/grasp_v2'/f'{a.checkpoint}_{a.mode}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
    command=[sys.executable,'-u',str(HERE/'autonomy_replay.py'),'play','--policy',str(ROOT/m['preserved_policy']),
             '--milestone',str(milestone),'--output',str(output)]
    for key,value in m['replay_settings'].items():
        if isinstance(value,bool):
            if value:command.append('--'+key)
        elif isinstance(value,list):command.extend(['--'+key,*map(str,value)])
        else:command.extend(['--'+key,str(value)])
    command.extend(['--headless','--video'] if a.mode=='video' else ['--realtime'])
    command.extend(extra)
    os.chdir(ROOT);os.execv(sys.executable,command)


if __name__=='__main__':main()
