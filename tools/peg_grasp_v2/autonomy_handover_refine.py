"""Collect measured handover recoveries, refine receiver BC, verify regressions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from autonomy_motion_curriculum import distribution_gate, sha
from preserve_autonomy import verify_archive

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def patch_collection_teacher(source):
    pressure="        delta=torch.where(f.amin(-1)<minimum,-close*DT,torch.where(f.amin(-1)>maximum,.05*DT,0.))"
    balancing="        balancing=(stage==10)&(f.amax(-1)>.10)&(imbalance.abs()>.2)"
    if source.count(pressure)!=1 or source.count(balancing)!=1:
        raise ValueError('Unknown archived collection teacher; refusing blind changes')
    source=source.replace(pressure,pressure+"\n        # Collection-only overload relief; never applied to neural inference.\n        if arm==1:delta=torch.where(f.amax(-1)>.65,torch.full_like(delta,.05*DT),delta)")
    return source.replace(balancing,"        balancing=((stage==10)|(stage==11))&(f.amax(-1)>.10)&(imbalance.abs()>.2)")


def patch_collection_main(source):
    use='                        use=learner_execution_mask(vec.age,teacher.stage,args.learner_fraction,args.learner_until_stage,args.learner_prefix_steps)'
    handoff='teacher.begin_receiver_recovery(vec.reference,previous_learner&~use)'
    chunk='                    chunks.append(dict(obs=x.clone(),actions=label.clone(),stage=stage,episode=episode_id.clone()))'
    metadata="source_hashes=manifest['source_hashes'])"
    for marker in (use,handoff,chunk,metadata):
        if source.count(marker)!=1:raise ValueError('Unknown collection ownership implementation')
    source=source.replace(use,use+"\n                        if args.receiver_recovery_handoff:\n                            from autonomy_recovery_mask import recovery_execution_mask\n                            ready=vec.progress.flags[:,1]&teacher.strong()[:,0]\n                            use=recovery_execution_mask(use,previous_learner,episode_id<args.episodes,ready)")
    source=source.replace(handoff,'teacher.begin_receiver_recovery(vec.reference,previous_learner&~use&(episode_id<args.episodes))')
    source=source.replace(chunk,chunk+"\n                    if args.receiver_recovery_handoff:chunks[-1]['expert_executed']=(~use)&(episode_id<args.episodes)")
    return source.replace(metadata,"source_hashes=manifest['source_hashes'],expert_execution_mask_recorded=('expert_executed' in data))")


def freeze_collection_model(parent, output):
    manifest=json.loads((parent.parent/'manifest.json').read_text())
    if sha(parent)!=manifest['policy_sha256']:raise ValueError('Changed parent weights')
    for name,digest in manifest['source_hashes'].items():
        if sha(parent.parent/'sources'/name)!=digest:raise ValueError('Changed parent source')
    shutil.copytree(parent.parent,output,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    path=output/'sources/autonomy_full_teacher.py'
    path.write_text(patch_collection_teacher(path.read_text()))
    main=output/'sources/autonomy_main.py';main.write_text(patch_collection_main(main.read_text()))
    shutil.copy2(HERE/'autonomy_recovery_mask.py',output/'sources/autonomy_recovery_mask.py')
    changed=[name for name,digest in manifest['source_hashes'].items() if sha(output/'sources'/name)!=digest]
    if set(changed)!={'autonomy_full_teacher.py','autonomy_main.py'} or sha(output/'policy.pt')!=sha(parent):
        raise ValueError('Unexpected inference or physics change')
    manifest['source_hashes']['autonomy_full_teacher.py']=sha(path)
    manifest['source_hashes']['autonomy_main.py']=sha(main)
    manifest['source_hashes']['autonomy_recovery_mask.py']=sha(output/'sources/autonomy_recovery_mask.py')
    manifest.update(parent_policy_sha256=sha(parent),collection_teacher_change=dict(
        receiver_peak_relief_n=.65,receiver_relief_rate_rad_s=.05,balance_during_donor_release=True,
        inference_unchanged=True,object_pose_writes=False))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return output/'policy.pt'


def paired_retention(reference,candidate):
    a,b=reference.get('cases',[]),candidate.get('cases',[])
    if reference.get('seed')!=candidate.get('seed') or not a or len(a)!=len(b):return False
    for old,new in zip(a,b):
        if any(old.get('reset',{}).get(k)!=new.get('reset',{}).get(k)
               for k in ('source','target','jitter_mm','yaw_deg')):return False
        if old.get('success') and not new.get('success'):return False
    return True


def read_physical(out,policy,seed):
    path=out/'evaluation.json'
    if not path.is_file():
        error=out/'error.json'
        raise RuntimeError('Physical evaluation did not complete: '+(error.read_text() if error.exists() else str(out)))
    report=json.loads(path.read_text())
    close=json.loads((out/'environment_close.json').read_text())
    if (report.get('complete') is not True or report.get('seed')!=seed
        or report.get('policy_sha256')!=sha(policy) or close.get('environment_closed') is not True
        or close.get('evaluation_sha256')!=sha(path)):
        raise ValueError('Unverified physical result, seed, weights or cleanup')
    speed=report.get('motion_reference',{})
    if speed.get('profile')!='travel' or speed.get('free_scale')!=2 or speed.get('transport_scale')!=2 or speed.get('contact_scale')!=1:
        raise ValueError('Speed profile changed')
    return report


def preserve_candidate(output,policy,state):
    """Save only after physical gates and a complete successful front video."""
    name='checkpoint12-assisted-handover-bc'
    bundle=ROOT/'.checkpoint_runs'/name
    milestone=ROOT/'milestones'/f'{name}.json'
    destination=ROOT/'logs/grasp_v2/videos/checkpoint12.mp4'
    if bundle.exists() or milestone.exists() or destination.exists():raise ValueError('Checkpoint12 already exists')
    video_out=output/'front_video'
    report=read_physical(video_out,policy,2468)
    if not distribution_gate(report,.5,5.,1,1.):raise ValueError('Checkpoint video did not succeed')
    video=video_out/'inference_front.mp4'
    media=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=codec_name,width,height,avg_frame_rate,nb_frames,duration','-of','json',str(video)],text=True))['streams'][0]
    duration=float(media.get('duration',0));frames=int(media.get('nb_frames',0))
    if not frames or abs(duration-report['cases'][0]['length']*.01)>.05:raise ValueError('Incomplete video')
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
    bundle.mkdir(parents=True)
    shutil.copytree(policy.parent,bundle/'model',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    reports=bundle/'reports';reports.mkdir()
    for row in state['results']:
        out=Path(row['report']).parent
        for file in ('evaluation.json','environment_close.json'):
            shutil.copy2(out/file,reports/f'{row["label"]}_{file}')
    collection=Path(state['collection_dataset']).parent
    for file in ('evaluation.json','environment_close.json','completed_episodes.json','error.json'):
        if (collection/file).is_file():shutil.copy2(collection/file,reports/f'collection_{file}')
    for file in ('evaluation.json','environment_close.json'):
        shutil.copy2(video_out/file,reports/f'video_{file}')
    shutil.copy2(output/'workflow_source.py',reports/'workflow_source.py')
    shutil.copy2(output/'fit_source.py',reports/'fit_source.py')
    (bundle/'videos').mkdir();shutil.copy2(video,bundle/'videos/checkpoint12.mp4')
    replay=dict(skill='full',num_envs=1,episodes=1,episode_steps=10000,seed=2468,sources=['L5'],targets=['R2'],
        jitter_mm=.5,yaw_deg=5.,grasp_support=1.,support_profile='firm',placement_support=True,
        motion_profile='travel',free_speed_scale=2.,transport_speed_scale=2.)
    result=dict(name=name,date=datetime.now(timezone.utc).date().isoformat(),
        algorithm='Receiver-only Behavior Cloning refinement; not SAC',
        preserved_bundle=str(bundle.relative_to(ROOT)),preserved_policy=str((bundle/'model/policy.pt').relative_to(ROOT)),
        policy_sha256=sha(policy),physics_manifest='milestones/checkpoint3-full-assisted.json',
        physics_manifest_sha256=sha(ROOT/'milestones/checkpoint3-full-assisted.json'),
        reference_checkpoint='milestones/checkpoint11-assisted-full-bc-yaw5.json',
        replay_settings=replay,assisted_grasp=True,native_full_policy_verified=False,
        teacher_at_inference=False,programmed_task_controller=False,
        training=json.loads((policy.parent/'training.json').read_text()),
        verified_results=state['results'],video='logs/grasp_v2/videos/checkpoint12.mp4',
        limitations=['Firm grasp spring and free-placement guide remain enabled',
                     'Only L5 to R2, same-post jitter0.5mm and yaw5deg; no larger domains certified'],
        files={str(p.relative_to(bundle)):sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()})
    with milestone.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    verify_archive(milestone)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with video.open('rb') as source,destination.open('xb') as target:shutil.copyfileobj(source,target)
    for suffix,source in (('.evaluation.json',video_out/'evaluation.json'),('.environment_close.json',video_out/'environment_close.json')):
        with destination.with_suffix(suffix).open('x') as stream:stream.write(source.read_text())
    metadata=dict(checkpoint='checkpoint12',milestone=str(milestone.relative_to(ROOT)),policy_sha256=sha(policy),
        video_sha256=sha(destination),evaluation_sha256=sha(video_out/'evaluation.json'),episode_success=True,
        assisted_grasp=True,media=media,encoder='h264_nvenc',camera=dict(eye=[.025,.32,.25],target=[.025,0,.035]))
    with destination.with_suffix('.json').open('x') as stream:json.dump(metadata,stream,indent=2);stream.write('\n')
    with (ROOT/'README.md').open('a') as stream:
        stream.write('\n## Checkpoint12: handover BC corretto, presa assistita\n\n'
            'Affinata la pinza di PSM2 su nuove dimostrazioni; gli altri pesi sono conservati. '
            'Verifiche fisiche e video in [checkpoint12](milestones/checkpoint12-assisted-handover-bc.json). '
            'Presa assistita, stesso cilindro L5 e target R2, offset ±0,5 mm e rotazione ±5°.\n\n'
            '```bash\nbash peg_grasp_v2.sh checkpoint12-play --episodes 3\n'
            '# Dal Mac, fuori dalla sessione SSH:\n'
            'scp francesco@100.77.29.104:/home/francesco/Documents/GIGI/i4h-workflows/logs/grasp_v2/videos/checkpoint12.mp4 ~/Downloads/\n'
            'open ~/Downloads/checkpoint12.mp4\n```\n')
    return str(milestone.relative_to(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=32);p.add_argument('--collection_seed',type=int,default=55001)
    p.add_argument('--completed_recoveries',type=Path,help='Explicitly reuse verified full trajectories from an interrupted collection')
    args=p.parse_args()
    if args.episodes<16:raise ValueError('At least16 recovery episodes required')
    archive=verify_archive(ROOT/'milestones/checkpoint11-assisted-full-bc-yaw5.json')
    parent=ROOT/archive['preserved_policy']
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,args.output/'workflow_source.py')
    shutil.copy2(HERE/'autonomy_receiver_fit.py',args.output/'fit_source.py')
    state=dict(complete=False,stage='freeze_collection_runtime',algorithm='Expert recoveries then receiver-only BC refinement; not SAC',
        parent_checkpoint='checkpoint11',parent_policy_sha256=sha(parent),results=[],
        assisted_grasp=True,native_full_policy_verified=False,training_running=False,
        condition=dict(source='L5',target='R2',jitter_mm=.5,yaw_deg=5.),
        teacher_at_inference=False,teacher_collection_only=True,
        inference_motor_router_donor_and_pose_weights_protected=True,
        fit_settings=dict(updates=2000,batch_size=1024,learning_rate=2e-5,reference_kl_weight=.1),
        validation_seeds=[2468,54321],collection_seed=args.collection_seed)
    state['workflow_source_hashes']={name:sha(HERE/name) for name in (
        'autonomy_handover_refine.py','autonomy_receiver_fit.py','autonomy_replay.py',
        'autonomy_learning.py','autonomy_motion_curriculum.py','autonomy_constraints_gate.py',
        'preserve_autonomy.py','geometry.py','rl_core.py','autonomy_recovery_mask.py')}
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        tmp=args.output/'status.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(args.output/'status.json')
    def run(module,argv,label):
        for name,digest in state['workflow_source_hashes'].items():
            if sha(HERE/name)!=digest:raise ValueError(f'Workflow changed during execution: {name}')
        state['stage']=label;save();print(f'[HANDOVER REFINE] {label}',flush=True)
        with (args.output/f'{label}.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(HERE/module),*map(str,argv)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save()
            try:code=child.wait()
            except BaseException:child.send_signal(signal.SIGINT);child.wait();raise
        state.pop('child_pid',None);save()
        if code not in (0,1):raise RuntimeError(f'{label} exited {code}')
        return code
    common=['--skill','full','--episode_steps',10000,'--sources','L5','--targets','R2',
        '--grasp_support',1,'--support_profile','firm','--placement_support',
        '--motion_profile','travel','--free_speed_scale',2,'--transport_speed_scale',2,'--headless']
    try:
        save();runtime=freeze_collection_model(parent,args.output/'collection_runtime')
        if args.completed_recoveries:
            dataset=args.completed_recoveries.resolve()
            from autonomy_receiver_fit import completed_trajectory_receipt
            import numpy as np
            with np.load(dataset,allow_pickle=False) as file:metadata=json.loads(str(file['metadata']))
            receipt=completed_trajectory_receipt(dataset,metadata)
            source_manifest=json.loads((dataset.parent/'manifest.json').read_text())
            runtime_manifest=json.loads((runtime.parent/'manifest.json').read_text())
            if (source_manifest.get('policy_sha256')!=sha(parent)
                or source_manifest['source_hashes']['autonomy_full_teacher.py']!=runtime_manifest['source_hashes']['autonomy_full_teacher.py']):
                raise ValueError('Completed recoveries differ from the parent or corrected collection teacher')
            state.update(collection_success=len(receipt['cases']),collection_episodes=receipt['observed_episodes'],
                source_collection_complete=False,source_requested_episodes=receipt['requested_episodes'],
                source_receipt_sha256=receipt['receipt_sha256'],source_closure_sha256=receipt['closure_sha256'])
        else:
            out=args.output/'recoveries'
            run('autonomy_replay.py',['collect','--policy',runtime,'--output',out,'--num_envs',16,'--episodes',args.episodes,
            '--seed',args.collection_seed,'--jitter_mm',.5,'--yaw_deg',5,'--learner_prefix_steps',2100,
            '--sync_learner_reference','--receiver_recovery_handoff','--receiver_mode','wall',
            '--receiver_wall_angle',300,'--receiver_wall_depth',4,'--receiver_wall_tilt',0,'--receiver_open_angle',.25,
            '--receiver_release_bottom',.03,'--receiver_pressure_min',.36,'--receiver_pressure_max',.42,
            '--donor_release_rate',.02,'--fixed_donor_during_contact','--recapture_receiver_after_transfer',*common],
                'collect_handover_recoveries')
            report=read_physical(out,runtime,args.collection_seed)
            state.update(collection_success=report['success'],collection_episodes=report['episodes'],source_collection_complete=True)
            dataset=out/'demonstrations.npz'
        state['collection_dataset']=str(dataset);state['collection_dataset_sha256']=sha(dataset) if dataset.is_file() else None;save()
        if state['collection_success']<8 or not dataset.is_file():
            state.update(complete=True,stage='insufficient_successful_recoveries');save();return 1
        state['training_running']=True;save()
        candidate=args.output/'receiver_bc'
        extra=['--allow_completed_receipt'] if args.completed_recoveries else []
        run('autonomy_receiver_fit.py',['--policy',runtime,'--dataset',dataset,'--output',candidate,
            '--updates',2000,'--refine_existing','--preserve_parent_sources','--learning_rate',2e-5,'--regularization',.1,*extra],
            'receiver_bc_training')
        state['training_running']=False;policy=candidate/'policy.pt';state['candidate_policy']=str(policy);state['candidate_policy_sha256']=sha(policy);save()
        old10=ROOT/'.checkpoint_runs/checkpoint10-assisted-full-bc-transit/reports'
        old11=ROOT/'.checkpoint_runs/checkpoint11-assisted-full-bc-yaw5/reports'
        conditions=[('fixed_pilot',3,123,0.,0.,None),
            ('yaw2_seed2468',16,2468,.5,2.,old10/'transit0_seed2468.json'),
            ('yaw2_seed54321',16,54321,.5,2.,old10/'transit0_seed54321.json'),
            ('yaw5_seed2468',16,2468,.5,5.,old11/'yaw5_seed2468.json'),
            ('yaw5_seed54321',16,54321,.5,5.,old11/'yaw5_seed54321.json')]
        verify_archive(ROOT/'milestones/checkpoint10-assisted-full-bc-transit.json')
        for label,count,seed,jitter,yaw,reference in conditions:
            out=args.output/label
            run('autonomy_replay.py',['evaluate','--policy',policy,'--output',out,'--num_envs',count,'--episodes',count,
                '--seed',seed,'--jitter_mm',jitter,'--yaw_deg',yaw,'--diagnostic_events','--diagnostic_interval',25,*common],label)
            report=read_physical(out,policy,seed)
            retained=reference is None or paired_retention(json.loads(reference.read_text()),report)
            passed=distribution_gate(report,jitter,yaw,count,1. if count==3 else 15/16) and retained
            state['results'].append(dict(label=label,success=report['success'],episodes=count,seed=seed,
                jitter_mm=jitter,yaw_deg=yaw,passed=passed,previous_successes_retained=retained,
                report=str(out/'evaluation.json'),report_sha256=sha(out/'evaluation.json')));save()
            if not passed:
                state.update(complete=True,stage='candidate_not_promoted',failed_condition=label,
                    reason='Keep checkpoint11; inspect physical failures before another fit or wider resets');save();return 1
        out=args.output/'front_video'
        run('autonomy_replay.py',['play','--policy',policy,'--output',out,'--num_envs',1,'--episodes',1,
            '--seed',2468,'--jitter_mm',.5,'--yaw_deg',5,'--video',*common],'record_candidate_front_video')
        state['milestone']=preserve_candidate(args.output,policy,state)
        state.update(complete=True,stage='checkpoint12_preserved_with_video',
                     next_step='Verified same-post yaw10 curriculum; larger ranges and new posts remain untested');save();return 0
    except BaseException as error:
        state.update(complete=False,training_running=False,
            stage='interrupted' if isinstance(error,KeyboardInterrupt) else 'workflow_error',error=f'{type(error).__name__}: {error}');save();raise


if __name__=='__main__':raise SystemExit(main())
