"""Matched GPU speed experiments on checkpoint12; preserve only verified results."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from autonomy_three_posts_certify import PAIRS, physical_report, media
from autonomy_three_posts import sha
from autonomy_motion_curriculum import speed_comparison
from preserve_autonomy import verify_archive

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'logs/grasp_v2/three_posts_pickup_bc_v60'


def preserve(output, policy, selected, rows, video_out):
    if not (output/'cuda_tests.log').read_text().rstrip().endswith('OK'):
        raise ValueError('CUDA control checks failed')
    name='checkpoint13-three-posts-fast-assisted'
    bundle=ROOT/'.checkpoint_runs'/name
    milestone=ROOT/'milestones'/f'{name}.json'
    videos=ROOT/'logs/grasp_v2/videos'
    first=videos/'checkpoint13.mp4';triple=videos/'checkpoint13_three_episodes.mp4'
    if any(p.exists() for p in (bundle,milestone,first,triple)):
        raise ValueError('Refuse to overwrite any saved checkpoint or video')
    report, passed=physical_report(video_out,policy,1)
    if not passed:raise ValueError('Video must demonstrate all three full physical transfers')
    source=video_out/'inference_front.mp4'
    full_media=media(source,sum(c['length'] for c in report['cases'])*.01)
    cut=output/'checkpoint13_first_episode.mp4'
    subprocess.run(['ffmpeg','-v','error','-n','-i',str(source),'-frames:v',str(report['cases'][0]['length']//4),
        '-an','-c:v','h264_nvenc','-preset','p4','-cq','22','-pix_fmt','yuv420p','-movflags','+faststart',str(cut)],check=True)
    first_media=media(cut,report['cases'][0]['length']*.01)
    shutil.copytree(policy.parent,bundle/'model',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (bundle/'reports').mkdir();(bundle/'videos').mkdir()
    for row in rows:
        folder=Path(row['report']).parent
        for file in ('evaluation.json','environment_close.json'):
            shutil.copy2(folder/file,bundle/'reports'/f'{row["label"]}_{file}')
    shutil.copy2(__file__,bundle/'reports/speed_certify_source.py')
    for file in ('autonomy_speed_snapshot.py','test_fast_transit.py'):
        shutil.copy2(HERE/file,bundle/'reports'/file)
    shutil.copy2(output/'cuda_tests.log',bundle/'reports/cuda_tests.log')
    shutil.copy2(cut,bundle/'videos/checkpoint13.mp4')
    shutil.copy2(source,bundle/'videos/checkpoint13_three_episodes.mp4')
    replay=dict(skill='full',num_envs=1,episodes=3,episode_steps=10000,seed=59003,
        sources=[s for s,t in PAIRS],targets=[t for s,t in PAIRS],paired_posts=True,jitter_mm=0.,yaw_deg=0.,
        grasp_support=1.,support_profile='firm',placement_support=True,gripper_load_guard=True,
        upright_grasp_support=True,motion_profile='fast_travel',free_speed_scale=selected[0],transport_speed_scale=selected[1])
    m=dict(name=name,date=datetime.now(timezone.utc).date().isoformat(),algorithm='Checkpoint12 frozen BC actor; measured-clearance transit speed optimization, no retraining',
        preserved_bundle=str(bundle.relative_to(ROOT)),preserved_policy=str((bundle/'model/policy.pt').relative_to(ROOT)),
        policy_sha256=sha(policy),reference_checkpoint='milestones/checkpoint12-three-posts-assisted.json',
        physics_manifest='milestones/checkpoint3-full-assisted.json',physics_manifest_sha256=sha(ROOT/'milestones/checkpoint3-full-assisted.json'),
        replay_settings=replay,neural_weights_unchanged=True,assisted_grasp=True,native_full_policy_verified=False,
        teacher_at_inference=False,programmed_task_controller=False,verified_results=rows,
        motion_reference=report['motion_reference'],video=str(first.relative_to(ROOT)),three_episode_video=str(triple.relative_to(ROOT)),
        limitations=['Fastest passing profile among tested candidates, not a certified mechanical maximum',
            'Same three post pairs and small 0.1mm/0.5deg confirmation; explicit grasp and placement assistance remains active'],
        files={str(p.relative_to(bundle)):sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()})
    with milestone.open('x') as f:json.dump(m,f,indent=2);f.write('\n')
    verify_archive(milestone)
    for dest,src,details,indices in ((first,cut,first_media,[0]),(triple,source,full_media,[0,1,2])):
        with dest.open('xb') as f,src.open('rb') as s:shutil.copyfileobj(s,f)
        metadata=dict(checkpoint='checkpoint13',policy_sha256=sha(policy),video_sha256=sha(dest),
            evaluation_sha256=sha(video_out/'evaluation.json'),episode_indices=indices,physical_report_covers_episodes=3,
            all_three_episodes_success=True,assisted_grasp=True,media=details,replay_settings=replay)
        with dest.with_suffix('.json').open('x') as f:json.dump(metadata,f,indent=2);f.write('\n')
        for file,suffix in (('evaluation.json','.evaluation.json'),('environment_close.json','.environment_close.json')):
            with dest.with_suffix(suffix).open('x') as f:f.write((video_out/file).read_text())
    return str(milestone.relative_to(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait_pid',type=int,help='Existing first pilot process; wait for its verified closure')
    a=p.parse_args();a.output=a.output.resolve();policy=a.output/'frozen_model/policy.pt'
    verify_archive(ROOT/'milestones/checkpoint12-three-posts-assisted.json')
    tests=(a.output/'cuda_tests.log').read_text()
    if not tests.rstrip().endswith('OK'):raise ValueError('CUDA control tests must pass first')
    refs=dict(pilot=json.loads((BASE/'full_contact_pilot3/evaluation.json').read_text()),
        confirmation=json.loads((BASE/'certification/confirmation12/evaluation.json').read_text()))
    state=dict(complete=False,stage='pilot_sweep',policy_sha256=sha(policy),algorithm='Frozen BC matched speed sweep',
        candidates=[[4,4],[12,32],[8,16],[4,8],[2,16]],results=[],assisted_grasp=True,native_full_policy_verified=False)
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        tmp=a.output/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.output/'status.json')
    def record(folder,label,scales,count,jitter,yaw,reference=None):
        r,ok=physical_report(folder,policy,count//3,jitter,yaw)
        settings=r['motion_reference']
        if settings['free_scale']!=scales[0] or settings['transport_scale']!=scales[1] or settings['contact_scale']!=1:
            raise ValueError('Reported speed differs from tested configuration')
        matched=ok and (reference is None or speed_comparison(reference,r))
        old=sum(c['length'] for c in reference['cases'])/count*.01 if reference else None
        mean=sum(c['length'] for c in r['cases'])/count*.01
        state['results'].append(dict(label=label,scales=list(scales),success=r['success'],episodes=count,
            passed=ok,matched_speed_accepted=matched,mean_episode_seconds=mean,baseline_mean_seconds=old,
            duration_reduction_percent=(1-mean/old)*100 if old else None,
            cases=[dict(source=c['reset']['source'],target=c['reset']['target'],seconds=c['length']*.01,
                        success=c['success'],peak_force_n=c['peak_force_n'],hole_error_mm=c['hole_error_mm']) for c in r['cases']],
            report=str(folder/'evaluation.json'),report_sha256=sha(folder/'evaluation.json')))
        save();return r,matched
    def run(label,scales,count,jitter,yaw,seed,video=False):
        state['stage']=label;save()
        folder=a.output/label
        if folder.exists():raise ValueError('Existing run must not be overwritten: '+str(folder))
        command=[sys.executable,'-u',str(HERE/'autonomy_replay.py'),'play' if video else 'evaluate',
            '--policy',str(policy),'--output',str(folder),'--skill','full','--num_envs','1' if video else str(count),
            '--episodes',str(count),'--episode_steps','10000','--sources',*[s for s,t in PAIRS],
            '--targets',*[t for s,t in PAIRS],'--paired_posts','--jitter_mm',str(jitter),'--yaw_deg',str(yaw),
            '--seed',str(seed),'--grasp_support','1','--support_profile','firm','--placement_support',
            '--gripper_load_guard','--upright_grasp_support','--motion_profile','fast_travel',
            '--free_speed_scale',str(scales[0]),'--transport_speed_scale',str(scales[1]),
            '--diagnostic_events','--headless']+(['--video'] if video else [])
        with (a.output/f'{label}.log').open('x') as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save();code=child.wait()
        state.pop('child_pid',None);save()
        if code not in (0,1):raise RuntimeError(f'{label}: process exited {code}')
        return folder
    try:
        save()
        if a.wait_pid:
            state.update(stage='await_initial_pilot',child_pid=a.wait_pid);save()
            while not (a.output/'travel4_pilot/environment_close.json').exists():
                try:os.kill(a.wait_pid,0)
                except ProcessLookupError:raise RuntimeError('Initial pilot ended without verified closure')
                time.sleep(2)
            state.pop('child_pid',None)
        _,fallback=record(a.output/'travel4_pilot','pilot_f4_t4',(4,4),3,0,0,refs['pilot'])
        # Test the highest caps first; descend after a physical rejection.
        # Do not spend more trials once a full matched confirmation succeeds.
        candidates=[(12,32),(8,16),(4,8),(2,16)]+([(4,4)] if fallback else [])
        for scales in candidates:
            if scales!=(4,4):
                label=f'pilot_f{scales[0]}_t{scales[1]}'
                folder=run(label,scales,3,0,0,59001)
                _,ok=record(folder,label,scales,3,0,0,refs['pilot'])
                if not ok:continue
            label=f'confirm_f{scales[0]}_t{scales[1]}'
            folder=run(label,scales,12,.1,.5,59002)
            _,ok=record(folder,label,scales,12,.1,.5,refs['confirmation'])
            if not ok:continue
            folder=run(f'video_f{scales[0]}_t{scales[1]}',scales,3,0,0,59003,True)
            _,ok=record(folder,'three_episode_front_video',scales,3,0,0)
            if not ok:continue
            state['selected_scales']=list(scales)
            state['milestone']=preserve(a.output,policy,scales,state['results'],folder)
            state.update(complete=True,stage='checkpoint13_saved_video_needs_visual_review');save();return 0
        state.update(complete=True,stage='no_faster_profile_passed');save();return 1
    except BaseException as e:
        state.update(complete=False,stage='workflow_error',error=f'{type(e).__name__}: {e}');save();raise


if __name__=='__main__':raise SystemExit(main())
