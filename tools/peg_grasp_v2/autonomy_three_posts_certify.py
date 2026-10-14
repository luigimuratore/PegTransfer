"""Validate three learned post transfers and preserve their exact front videos."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from autonomy_three_posts import sha
from preserve_autonomy import verify_archive

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PAIRS=[('L5','R2'),('L6','R1'),('L4','R3')]

def physical_report(folder,policy,repeat,jitter=0.,yaw=0.):
    path=folder/'evaluation.json'
    if not path.exists():raise ValueError('Simulation incomplete: '+str(folder))
    r=json.loads(path.read_text());close=json.loads((folder/'environment_close.json').read_text())
    if (not r.get('complete') or r.get('episodes')!=3*repeat or r.get('policy_sha256')!=sha(policy)
        or close.get('environment_closed') is not True or close.get('evaluation_sha256')!=sha(path)
        or r.get('teacher_at_inference') is not False or r.get('programmed_task_controller') is not False
        or r.get('assisted_grasp') is not True):raise ValueError('Unverified physical report or closure')
    distribution=r['reset_distribution']
    if (distribution.get('pair_order')!=[list(p) for p in PAIRS] or distribution.get('paired_posts') is not True
        or distribution.get('jitter_mm')!=jitter or distribution.get('yaw_deg')!=yaw):
        raise ValueError('Different post/reset schedule')
    seen=Counter((c['reset']['source'],c['reset']['target']) for c in r['cases'])
    if seen!=Counter({p:repeat for p in PAIRS}):raise ValueError('Missing or repeated pair')
    passed=r['success']==3*repeat
    for c in r['cases']:
        support=c.get('grasp_support',{})
        passed &= all(c.get(k) is True for k in ('success','full_success','grasp_psm1','lift_stable','grasp_psm2','handover'))
        passed &= (c.get('failure') is False and c.get('timeout') is False and c['peak_force_n']<=1
                   and c['hole_error_mm']<1.5 and support.get('holder')==0 and support.get('placement_guide_active') is False)
    guard=r.get('gripper_load_guard',{});upright=r.get('upright_grasp_support',{})
    ceilings=guard.get('ceiling_n',[])
    if (len(ceilings)!=2 or any(abs(a-b)>1e-6 for a,b in zip(ceilings,[.85,.65]))
        or upright.get('torque_limit_nm')!=.004):
        raise ValueError('Different explicit grasp constraints')
    return r,bool(passed)

def media(path,seconds):
    result=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=codec_name,width,height,avg_frame_rate,nb_frames,duration','-of','json',str(path)],text=True))['streams'][0]
    if int(result.get('nb_frames',0))<=0 or abs(float(result['duration'])-seconds)>.05:
        raise ValueError('Video duration differs from complete physical episode(s)')
    subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True)
    return result

def preserve(output,policy,results,video_out):
    name='checkpoint12-three-posts-assisted'
    bundle=ROOT/'.checkpoint_runs'/name;milestone=ROOT/'milestones'/f'{name}.json'
    canonical=ROOT/'logs/grasp_v2/videos/checkpoint12.mp4'
    triple=canonical.with_name('checkpoint12_three_episodes.mp4')
    if any(p.exists() for p in (bundle,milestone,canonical,triple)):raise ValueError('Refuse checkpoint/video overwrite')
    report,passed=physical_report(video_out,policy,1)
    if not passed:raise ValueError('Three-episode front replay did not succeed')
    video=video_out/'inference_front.mp4'
    all_media=media(video,sum(c['length'] for c in report['cases'])*.01)
    first=output/'checkpoint12_first_episode.mp4'
    frames=report['cases'][0]['length']//4
    subprocess.run(['ffmpeg','-v','error','-n','-i',str(video),'-frames:v',str(frames),
        '-an','-c:v','h264_nvenc','-preset','p4','-cq','22','-pix_fmt','yuv420p','-movflags','+faststart',str(first)],check=True)
    first_media=media(first,report['cases'][0]['length']*.01)
    bundle.mkdir(parents=True)
    shutil.copytree(policy.parent,bundle/'model',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (bundle/'reports').mkdir();(bundle/'videos').mkdir()
    for row in results:
        folder=Path(row['report']).parent
        for file in ('evaluation.json','environment_close.json'):
            shutil.copy2(folder/file,bundle/'reports'/f'{row["label"]}_{file}')
    shutil.copy2(__file__,bundle/'reports/certify_source.py')
    training=json.loads((policy.parent/'training.json').read_text())
    if training.get('dataset'):
        folder=Path(training['dataset']).parent
        for filename in ('evaluation.json','environment_close.json','manifest.json'):
            shutil.copy2(folder/filename,bundle/'reports'/f'lift_collection_{filename}')
    shutil.copy2(video,bundle/'videos/checkpoint12_three_episodes.mp4')
    shutil.copy2(first,bundle/'videos/checkpoint12.mp4')
    replay=dict(skill='full',num_envs=1,episodes=3,episode_steps=10000,seed=59003,
        sources=[p[0] for p in PAIRS],targets=[p[1] for p in PAIRS],paired_posts=True,jitter_mm=0.,yaw_deg=0.,
        grasp_support=1.,support_profile='firm',placement_support=True,gripper_load_guard=True,
        upright_grasp_support=True,motion_profile='travel',free_speed_scale=2.,transport_speed_scale=2.)
    m=dict(name=name,date=datetime.now(timezone.utc).date().isoformat(),
        algorithm=training['algorithm']+'; explicit temporary grasp constraints',
        preserved_bundle=str(bundle.relative_to(ROOT)),preserved_policy=str((bundle/'model/policy.pt').relative_to(ROOT)),
        policy_sha256=sha(policy),physics_manifest='milestones/checkpoint3-full-assisted.json',
        physics_manifest_sha256=sha(ROOT/'milestones/checkpoint3-full-assisted.json'),
        reference_checkpoint='milestones/checkpoint11-assisted-full-bc-yaw5.json',replay_settings=replay,
        assisted_grasp=True,native_full_policy_verified=False,teacher_at_inference=False,
        programmed_task_controller=False,training=training,
        neural_weights_unchanged=sha(policy)==json.loads((ROOT/'milestones/checkpoint11-assisted-full-bc-yaw5.json').read_text())['policy_sha256'],
        verified_results=results,video=str(canonical.relative_to(ROOT)),three_episode_video=str(triple.relative_to(ROOT)),
        limitations=['Three explicit pairs; small 0.1mm/0.5deg pilot, not broad domain-randomization certification',
            'Firm spring, earlier measured acquisition, upright torque, jaw load relief and placement guide are active'],
        files={str(p.relative_to(bundle)):sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()})
    with milestone.open('x') as f:json.dump(m,f,indent=2);f.write('\n')
    verify_archive(milestone)
    for destination,source,description,details in (
        (canonical,first,'First complete episode from the three-episode successful replay',first_media),
        (triple,video,'All three complete episodes; three different sources and targets',all_media)):
        with destination.open('xb') as f,source.open('rb') as s:shutil.copyfileobj(s,f)
        with destination.with_suffix('.json').open('x') as f:
            json.dump(dict(checkpoint='checkpoint12',policy_sha256=sha(policy),video_sha256=sha(destination),
                evaluation_sha256=sha(video_out/'evaluation.json'),description=description,
                episode_indices=[0] if destination==canonical else [0,1,2],all_three_episodes_success=True,
                physical_report_covers_episodes=3,assisted_grasp=True,media=details),f,indent=2);f.write('\n')
        for file,suffix in (('evaluation.json','.evaluation.json'),('environment_close.json','.environment_close.json')):
            with destination.with_suffix(suffix).open('x') as f:f.write((video_out/file).read_text())
    with (ROOT/'README.md').open('a') as f:
        f.write('\n## Checkpoint12: tre picking e tre placement, con presa assistita\n\n'
            'La rete BC trasferisce in ordine **L5→R2, L6→R1, L4→R3**. '
            'Conserva i controlli di movimento e pinze di checkpoint11, affina la scelta del lift e aggiunge vincoli temporanei alla presa; non è SAC. '
            'Nessun insegnante in inference. Risultati e aiuti dichiarati nel '
            '[report checkpoint12](milestones/checkpoint12-three-posts-assisted.json).\n\n'
            '```bash\n# Sulla workstation, nel terminale del desktop:\n'
            'bash peg_grasp_v2.sh checkpoint12-play\n'
            '# Dal Mac, fuori da SSH, scarica tutti e tre gli episodi:\n'
            'scp francesco@100.77.29.104:/home/francesco/Documents/GIGI/i4h-workflows/logs/grasp_v2/videos/checkpoint12_three_episodes.mp4 ~/Downloads/\n'
            'open ~/Downloads/checkpoint12_three_episodes.mp4\n```\n\n'
            'Il primo episodio completo è conservato anche in `logs/grasp_v2/videos/checkpoint12.mp4`.\n')
    return str(milestone.relative_to(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    # Confirmation and video both execute this archived snapshot. Later copies
    # into the named checkpoint preserve these exact bytes, not mutable sources.
    snapshot=a.output/'frozen_model'
    shutil.copytree(a.policy.parent,snapshot,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    if sha(snapshot/'policy.pt')!=sha(a.policy):raise ValueError('Archive copy changed weights')
    a.policy=snapshot/'policy.pt'
    state=dict(complete=False,stage='verify_pilot',results=[],policy_sha256=sha(a.policy),pairs=PAIRS,
               teacher_at_inference=False,assisted_grasp=True,native_full_policy_verified=False)
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        temp=a.output/'status.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(a.output/'status.json')
    def add(folder,label,repeat,jitter=0.,yaw=0.):
        r,passed=physical_report(folder,a.policy,repeat,jitter,yaw)
        state['results'].append(dict(label=label,success=r['success'],episodes=r['episodes'],passed=passed,
            report=str(folder/'evaluation.json'),report_sha256=sha(folder/'evaluation.json')));save()
        return passed
    def run(label,count,jitter,yaw,seed,video=False):
        state['stage']=label;save()
        command=[sys.executable,'-u',str(HERE/'autonomy_replay.py'),'play' if video else 'evaluate',
            '--policy',str(a.policy),'--output',str(a.output/label),'--skill','full','--num_envs','1' if video else str(count),
            '--episodes',str(count),'--episode_steps','10000','--sources',*[s for s,t in PAIRS],
            '--targets',*[t for s,t in PAIRS],'--paired_posts','--jitter_mm',str(jitter),'--yaw_deg',str(yaw),
            '--seed',str(seed),'--grasp_support','1','--support_profile','firm','--placement_support',
            '--gripper_load_guard','--upright_grasp_support','--motion_profile','travel','--free_speed_scale','2',
            '--transport_speed_scale','2','--headless']+(['--video'] if video else [])
        with (a.output/f'{label}.log').open('x') as f:
            child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save();code=child.wait()
        state.pop('child_pid',None);save()
        if code not in (0,1):raise RuntimeError(f'{label} exited {code}')
        return a.output/label
    try:
        save()
        if not add(a.pilot,'pilot',1):
            state.update(complete=True,stage='pilot_failed');save();return 1
        out=run('confirmation12',12,.1,.5,59002)
        if not add(out,'confirmation12',4,.1,.5):
            state.update(complete=True,stage='confirmation_failed');save();return 1
        out=run('three_episode_front_video',3,0.,0.,59003,True)
        if not add(out,'three_episode_front_video',1):
            state.update(complete=True,stage='video_failed');save();return 1
        state['milestone']=preserve(a.output,a.policy,state['results'],out)
        state.update(complete=True,stage='checkpoint12_preserved_with_three_episode_video');save();return 0
    except BaseException as e:
        state.update(complete=False,stage='workflow_error',error=f'{type(e).__name__}: {e}');save();raise

if __name__=='__main__':raise SystemExit(main())
