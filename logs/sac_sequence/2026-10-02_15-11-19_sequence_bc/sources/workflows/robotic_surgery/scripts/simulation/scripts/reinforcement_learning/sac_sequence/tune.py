"""Explicit user-run three-preset SAC pilot, selection and independent holdout."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from common import HERE, REPO, SKILLS, assets, checkpoint, fingerprint
from curriculum import PRESETS, passed

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline_report',type=Path,required=True)
    p.add_argument('--demonstrations',type=Path,required=True)
    p.add_argument('--initialize_from',type=Path,required=True)
    p.add_argument('--skill',choices=SKILLS,required=True)
    p.add_argument('--guidance',type=float,required=True)
    p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--source',choices=('L5','L6'),default='L5')
    p.add_argument('--steps',type=int,default=25000)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--num_envs',type=int,default=16)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if not 0<=args.guidance<=1 or min(args.steps,args.episodes,args.num_envs)<=0:
        p.error('Invalid pilot budget/guidance')
    baseline=json.loads(args.baseline_report.read_text())
    if not (baseline.get('fingerprint')==fingerprint() and baseline.get('assets')==assets() and baseline.get('source')==args.source
            and baseline.get('complete') and baseline.get('baseline_controller') and baseline.get('skill')=='full'
            and baseline.get('episodes',0)>=3 and baseline.get('full_success_rate')==1.):
        p.error('First verify the current complete analytic sequence')
    if checkpoint(args.initialize_from)['source'] != args.source: p.error('Initial checkpoint source differs')
    from model import load_demos
    _,demo=load_demos(args.demonstrations)
    if demo['source']!=args.source: p.error('Demo source differs')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    trials=[]
    shared=['--source',args.source,'--skill',args.skill,'--guidance',str(args.guidance),
            '--num_envs',str(args.num_envs),'--device',args.device,'--headless']
    for preset,settings in PRESETS.items():
        run_name='pilot_'+preset+'_'+args.output_dir.name
        before=set((REPO/'logs/sac_sequence').glob('*_'+run_name))
        command=[sys.executable,str(HERE/'train.py'),*shared,'--initialize_from',str(args.initialize_from.resolve()),
                 '--demonstrations',str(args.demonstrations.resolve()),'--steps',str(args.steps),
                 '--bc_steps','500','--seed','42','--run_name',run_name]
        for key,value in settings.items(): command+=['--'+key,str(value)]
        subprocess.run(command,check=True)
        created=set((REPO/'logs/sac_sequence').glob('*_'+run_name))-before
        if len(created)!=1: raise RuntimeError('Pilot did not create one concrete run')
        model=created.pop()/'final.zip';checkpoint(model)
        output=args.output_dir/(preset+'_validation.json')
        subprocess.run([sys.executable,str(HERE/'evaluate.py'),*shared,'--checkpoint',str(model),
                        '--episodes',str(args.episodes),'--seed','123','--output',str(output)],check=True)
        report=json.loads(output.read_text())
        if not report.get('complete'): raise RuntimeError('Incomplete pilot evaluation')
        trials.append(dict(preset=preset,settings=settings,checkpoint=str(model),report=str(output),
                           rate=report['skill_success_rate'],full_rate=report['full_success_rate'],failures=report['failures']))
        (args.output_dir/'trials.json').write_text(json.dumps(trials,indent=2)+'\n')
    best=max(trials,key=lambda trial:(trial['rate'],trial['full_rate'],-trial['failures']))
    holdout=args.output_dir/'selected_holdout_seed999.json'
    subprocess.run([sys.executable,str(HERE/'evaluate.py'),*shared,'--checkpoint',best['checkpoint'],
                    '--episodes',str(args.episodes),'--seed','999','--output',str(holdout)],check=True)
    validated=passed(json.loads(holdout.read_text()),.8,args.episodes)
    result=dict(fingerprint=fingerprint(),skill=args.skill,guidance=args.guidance,selected=best,
                holdout=str(holdout),promoted=validated,
                note='Selection applies only to this tested skill/guidance, not every phase or randomized source')
    (args.output_dir/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
