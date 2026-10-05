"""User-run gated curriculum. Never launch this implicitly from the coding agent."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from common import HERE, REPO, SKILLS, assets, checkpoint, fingerprint

PRESETS = {
    'conservative': dict(learning_rate=5e-5,bc_weight=10.,ent_coef=.0003,gradient_steps=1),
    'balanced': dict(learning_rate=1e-4,bc_weight=5.,ent_coef=.001,gradient_steps=1),
    'exploration': dict(learning_rate=1e-4,bc_weight=2.,ent_coef=.003,gradient_steps=1),
}

def passed(report, threshold, episodes):
    return (report.get('complete') is True and report.get('episodes',0)>=episodes
            and not report.get('baseline_controller',True)
            and report.get('skill_success_rate',0)>=threshold)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline_report',type=Path,required=True)
    p.add_argument('--demonstrations',type=Path,required=True)
    p.add_argument('--initialize_from',type=Path,required=True)
    p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--source',choices=('L5','L6'),default='L5')
    p.add_argument('--num_envs',type=int,default=16)
    p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--threshold',type=float,default=.8)
    p.add_argument('--preset',choices=tuple(PRESETS),default='conservative')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--eval_seed',type=int,default=123)
    p.add_argument('--holdout_seed',type=int,default=999)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if not 0<args.threshold<=1 or min(args.steps,args.episodes,args.num_envs)<=0:
        p.error('Invalid gate/budget')
    baseline=json.loads(args.baseline_report.read_text())
    if not (baseline.get('complete') and baseline.get('baseline_controller') and baseline.get('skill')=='full'
            and baseline.get('episodes',0)>=3 and baseline.get('full_success_rate')==1.
            and baseline.get('source')==args.source and baseline.get('fingerprint')==fingerprint()
            and baseline.get('assets')==assets()):
        p.error('First verify at least three successful full analytic baseline episodes on the current source/code')
    from model import load_demos
    _,meta=load_demos(args.demonstrations)
    if meta['source']!=args.source or 0. not in meta['guidance_levels']:
        p.error('Need current full demos on this source, including guidance=0 teacher actions')
    current=args.initialize_from.resolve()
    if checkpoint(current)['source'] != args.source: p.error('Initial checkpoint source differs')
    progress_path=args.output_dir/'curriculum.json'
    plan=[(skill,guidance) for skill in SKILLS for guidance in (1.,.5,.25,0.)]
    progress=dict(fingerprint=fingerprint(),source=args.source,preset=args.preset,steps=args.steps,
                  episodes=args.episodes,threshold=args.threshold,records=[],holdouts=[],complete=False,
                  seed=args.seed,eval_seed=args.eval_seed,holdout_seed=args.holdout_seed,
                  num_envs=args.num_envs,device=args.device,
                  demonstration_sha256=meta['sha256'])
    if args.resume:
        progress=json.loads(progress_path.read_text())
        for key in ('fingerprint','source','preset','steps','episodes','threshold','seed','eval_seed',
                    'holdout_seed','num_envs','device'):
            expected=fingerprint() if key=='fingerprint' else getattr(args,key)
            if progress[key]!=expected: p.error('Resume configuration/code differs from previous curriculum')
        if progress['demonstration_sha256'] != meta['sha256']:
            p.error('Resume requires the same demonstration archive')
        successful=[r for r in progress['records'] if r['passed']]
        if progress['records']: current=Path(progress['records'][-1]['checkpoint'])
        checkpoint(current)
        index=len(successful)
    else:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        index=0
    for index in range(index,len(plan)):
        skill,guidance=plan[index]
        attempt=sum(r['index']==index for r in progress['records'])
        label=f'{index:02d}_{skill}_g{guidance:g}_attempt{attempt}'
        output=args.output_dir/(label+'.json')
        shared=['--source',args.source,'--num_envs',str(args.num_envs),'--device',args.device,'--headless']
        evaluation=[sys.executable,str(HERE/'evaluate.py'),*shared,'--checkpoint',str(current),
                    '--skill',skill,'--guidance',str(guidance),'--episodes',str(args.episodes),
                    '--seed',str(args.eval_seed),'--output',str(output)]
        subprocess.run(evaluation,check=True)
        report=json.loads(output.read_text())
        if not passed(report,args.threshold,args.episodes):
            run_name='curriculum_'+label
            before=set((REPO/'logs/sac_sequence').glob('*_'+run_name))
            training=[sys.executable,str(HERE/'train.py'),*shared,'--initialize_from',str(current),
                      '--skill',skill,'--guidance',str(guidance),'--demonstrations',str(args.demonstrations.resolve()),
                      '--steps',str(args.steps),'--bc_steps','500','--seed',str(args.seed),'--run_name',run_name]
            for key,value in PRESETS[args.preset].items(): training += ['--'+key,str(value)]
            subprocess.run(training,check=True)
            created=set((REPO/'logs/sac_sequence').glob('*_'+run_name))-before
            if len(created)!=1: raise RuntimeError('Training did not produce one concrete run; inspect stderr/Kit log')
            current=created.pop()/'final.zip'; checkpoint(current)
            output=args.output_dir/(label+'_trained.json')
            evaluation[evaluation.index('--checkpoint')+1]=str(current)
            evaluation[evaluation.index('--output')+1]=str(output)
            subprocess.run(evaluation,check=True);report=json.loads(output.read_text())
        gate=passed(report,args.threshold,args.episodes)
        progress['records'].append(dict(index=index,skill=skill,guidance=guidance,checkpoint=str(current),
            report=str(output),passed=gate,rate=report['skill_success_rate']))
        progress_path.write_text(json.dumps(progress,indent=2)+'\n')
        if not gate:
            print(f'[CURRICULUM] STOP: {skill}, guidance={guidance}, rate={report["skill_success_rate"]}; '
                  'inspect the report/trace before another budget',flush=True)
            return
    holdout=args.output_dir/f'full_unguided_holdout{len(progress["holdouts"])}_seed{args.holdout_seed}.json'
    subprocess.run([sys.executable,str(HERE/'evaluate.py'),'--source',args.source,'--num_envs',str(args.num_envs),
        '--device',args.device,'--headless','--checkpoint',str(current),'--skill','full','--guidance','0',
        '--episodes',str(args.episodes),'--seed',str(args.holdout_seed),'--output',str(holdout)],check=True)
    validated=passed(json.loads(holdout.read_text()),args.threshold,args.episodes)
    progress['holdouts'].append(dict(report=str(holdout),checkpoint=str(current),passed=validated))
    progress['complete']=validated
    progress_path.write_text(json.dumps(progress,indent=2)+'\n')
    if not validated:
        print(f'[CURRICULUM] STOP: independent full guidance=0 holdout failed: {holdout}',flush=True)
        return
    print(f'[CURRICULUM] All gates passed, including full guidance=0: {current}',flush=True)

if __name__=='__main__': main()
