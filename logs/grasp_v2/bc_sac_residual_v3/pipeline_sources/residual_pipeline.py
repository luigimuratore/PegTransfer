"""Physical regression gates, initial residual training, confirmation and video."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path.cwd().resolve()
HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n')


def validate_child_result(folder, mode, video=False):
    """Kit can terminate Python with code zero while an exception is pending."""
    folder=Path(folder)
    if (folder/'error.json').exists():
        error=json.loads((folder/'error.json').read_text())
        raise RuntimeError(f'{mode} stopped: {error.get("error", error)}')
    status=json.loads((folder/'status.json').read_text())
    if status.get('complete') is not True:
        raise RuntimeError(f'{mode} did not complete: {status.get("stage")}')
    if mode=='train':
        if status.get('stage')!='candidate_requires_physical_evaluation':
            raise RuntimeError(f'Training stopped before producing a candidate: {status.get("stage")}')
        manifest=json.loads((folder/'manifest.json').read_text())
        if sha(folder/'candidate.zip')!=manifest['sac_sha256']:
            raise ValueError('Training candidate missing or changed')
    else:
        report=json.loads((folder/'evaluation.json').read_text())
        if report.get('complete') is not True or report.get('episodes')!=report.get('requested_episodes'):
            raise RuntimeError('Physical evaluation incomplete')
        if video:
            metadata=json.loads((folder/'video.json').read_text())
            if metadata.get('decode_verified') is not True or metadata.get('complete_episode') is not True:
                raise RuntimeError('Full candidate video not verified')
            if sha(folder/'inference_front.mp4')!=metadata['video_sha256']:
                raise ValueError('Candidate video changed')
    closure=json.loads((folder/'environment_close.json').read_text())
    if closure.get('closed') is not True:raise RuntimeError('Environment cleanup not confirmed')


def comparison(before,after):
    baseline={r['episode_id']:r for r in before['episodes_detail']}
    candidate={r['episode_id']:r for r in after['episodes_detail']}
    return (before['complete'] and after['complete'] and len(baseline)==len(candidate)==12
            and all(baseline[i]['reset']==candidate[i]['reset'] for i in baseline)
            and all(candidate[i]['success'] for i in baseline if baseline[i]['success'])
            and after['success']>=11)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--transitions',type=int,default=12000)
    p.add_argument('--num_envs',type=int,default=6)
    p.add_argument('--baseline3',type=Path,help='Reuse the already completed zero-residual physical screen')
    p.add_argument('--held_jaw_guard',action='store_true',help='Use the explicit aperture support in baseline, training and confirmation')
    p.add_argument('--grasp_stiffness',type=float,default=300.,help='Bounded spring stiffness in N/m, identical in every comparison')
    p.add_argument('--preflight6',type=Path,help='Reuse the six complete zero-residual episodes of the failed training seed; wait for cleanup if still running')
    p.add_argument('--spring_scope',choices=('all','donor'),default='donor')
    args=p.parse_args()
    out=(args.output or ROOT/'logs/grasp_v2'/f'bc_sac_residual_{datetime.now():%Y%m%d_%H%M%S_%f}').resolve()
    out.mkdir(parents=True,exist_ok=False)
    sources=out/'pipeline_sources';sources.mkdir()
    for name in ('residual_pipeline.py','residual_main.py','residual_env.py','residual_learning.py','residual_grasp_guard.py','residual_contact_diagnostic.py','residual_collision_support.py','residual_spring_support.py'):
        shutil.copy2(HERE/name,sources/name)
    runner=sources/'residual_main.py'
    state=dict(complete=False,stage='starting',algorithm='Frozen checkpoint13 BC + residual SAC',
        latest_preserved_checkpoint='checkpoint13',new_checkpoint_promoted=False,
        pipeline_source_hashes={f.name:sha(f) for f in sources.iterdir()},
        transitions=args.transitions,results=[])
    state['held_jaw_guard_enabled']=args.held_jaw_guard
    state['grasp_stiffness_n_m']=args.grasp_stiffness
    state['spring_scope']=args.spring_scope
    def run(label,mode,extras):
        state['stage']=label;write(out/'status.json',state)
        command=[sys.executable,'-u',str(runner),mode,'--headless','--output',str(out/label),*extras]
        command.extend(['--grasp_stiffness',str(args.grasp_stiffness)])
        command.extend(['--spring_scope',args.spring_scope])
        if args.held_jaw_guard:command.append('--held_jaw_guard')
        write(out/f'{label}_command.json',command)
        with (out/f'{label}.log').open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
        validate_child_result(out/label,mode,video='--video' in extras)
        path=out/label/'evaluation.json'
        if path.exists():
            report=json.loads(path.read_text())
            state['results'].append(dict(label=label,report=str(path),sha256=sha(path),success=report['success'],episodes=report['episodes']))
            write(out/'status.json',state)
            return report
    try:
        if args.preflight6:
            folder=args.preflight6.resolve()
            state['stage']='waiting_for_preflight6';write(out/'status.json',state)
            deadline=time.monotonic()+1200
            while not (folder/'environment_close.json').exists():
                if (folder/'error.json').exists() or (folder/'interruption_audit.json').exists():
                    raise RuntimeError('Preflight6 failed or was interrupted')
                if time.monotonic()>deadline:raise RuntimeError('Preflight6 completion deadline exceeded')
                time.sleep(10)
            validate_child_result(folder,'baseline')
            path=folder/'evaluation.json';fixed=json.loads(path.read_text())
            parent=json.loads((ROOT/'milestones/checkpoint13-three-posts-fast-assisted.json').read_text())
            if (fixed['algorithm']!='Frozen BC, exactly zero residual' or fixed['seed']!=61001
                or fixed['base_policy_sha256']!=parent['policy_sha256'] or fixed['episodes']!=6 or fixed['success']!=6
                or fixed['reset_distribution']['jitter_mm']!=0 or fixed['reset_distribution']['yaw_deg']!=0
                or fixed.get('collision_support',{}).get('enabled',False)
                or fixed.get('held_jaw_guard',{}).get('enabled',False)!=args.held_jaw_guard
                or any(row['grasp_support']['stiffness_n_m']!=args.grasp_stiffness for row in fixed['episodes_detail'])
                or json.loads((folder/'provenance.json').read_text())['parameters'].get('spring_scope','all')!=args.spring_scope):
                raise ValueError('Preflight6 is not a matched successful zero-residual baseline')
            state['results'].append(dict(label='preflight6_reused',report=str(path),sha256=sha(path),success=6,episodes=6))
        elif args.baseline3:
            path=args.baseline3.resolve()/'evaluation.json'
            fixed=json.loads(path.read_text())
            parent=json.loads((ROOT/'milestones/checkpoint13-three-posts-fast-assisted.json').read_text())
            if (fixed['algorithm']!='Frozen BC, exactly zero residual' or fixed['seed']!=59003
                or fixed['base_policy_sha256']!=parent['policy_sha256'] or fixed['requested_episodes']!=3
                or fixed['reset_distribution']['jitter_mm']!=0 or fixed['reset_distribution']['yaw_deg']!=0
                or fixed.get('held_jaw_guard',{}).get('enabled',False)!=args.held_jaw_guard
                or (args.grasp_stiffness!=300 and any(row['grasp_support']['stiffness_n_m']!=args.grasp_stiffness for row in fixed['episodes_detail']))):
                raise ValueError('Incompatible cached physical baseline')
            state['results'].append(dict(label='baseline3_reused',report=str(path),sha256=sha(path),success=fixed['success'],episodes=fixed['episodes']))
        else:
            fixed=run('baseline3','baseline',['--num_envs','3','--episodes','3','--seed','59003'])
        expected=6 if args.preflight6 else 3
        if not fixed['complete'] or fixed['success']!=expected:raise RuntimeError('Fixed baseline gate failed')
        baseline=run('baseline12','baseline',['--num_envs','12','--episodes','12','--seed','59002','--jitter_mm','.1','--yaw_deg','.5'])
        if not baseline['complete'] or baseline['success']!=12:raise RuntimeError('Baseline12 gate failed')
        run('training','train',['--num_envs',str(args.num_envs),'--transitions',str(args.transitions),'--seed','61001'])
        checkpoint=out/'training/candidate.zip'
        candidate=run('confirmation12','evaluate',['--checkpoint',str(checkpoint),'--num_envs','12','--episodes','12',
                      '--seed','59002','--jitter_mm','.1','--yaw_deg','.5'])
        # Every saved experimental candidate receives a full frontal episode, even on regression.
        video=run('candidate_video','play',['--checkpoint',str(checkpoint),'--video','--num_envs','1','--episodes','1','--seed','59003'])
        tag='bc_sac_residual_'+out.name.removeprefix('bc_sac_residual_')
        video_path=ROOT/'logs/grasp_v2/videos'/f'{tag}.mp4'
        for source,target in ((out/'candidate_video/inference_front.mp4',video_path),
                              (out/'candidate_video/video.json',video_path.with_suffix('.json')),
                              (out/'candidate_video/evaluation.json',video_path.with_suffix('.evaluation.json'))):
            if target.exists():raise ValueError('A recorded candidate video must not be overwritten')
            with target.open('xb') as f:f.write(source.read_bytes())
        passed=comparison(baseline,candidate) and video['complete'] and video['success']==1
        state.update(complete=True,stage='pilot_passed' if passed else 'pilot_regression',
            candidate=str(checkpoint),candidate_sha256=sha(checkpoint),video=str(video_path),
            matched_baseline_preserved=comparison(baseline,candidate),
            new_checkpoint_promoted=False,physical_assistance_active=True,
            next_step='Independent seed confirmation before increasing residual/randomization or reducing assistance')
        write(out/'status.json',state)
    except Exception as error:
        state.update(complete=True,stage='stopped_on_failed_gate_or_error',error=repr(error))
        write(out/'status.json',state)
        raise


if __name__=='__main__':main()
