"""Frozen checkpoint11 with explicit paired post resets and three-episode video.

Only reset allocation and video/report plumbing change. The learned actor,
motor control, contact physics and success criteria are preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
POSTS_SOURCE="SOURCES={'L5':(0.,0.),'L6':(0.,-.030),'L4':(0.,.030),'L2':(-.042,0.),'L1':(-.042,.030),'L3':(-.042,-.030)}"
POSTS_TARGET="TARGETS={'R1':(.042,-.015),'R2':(.042,.015),'R3':(.084,.035),'R4':(.084,-.035),'R5':(.126,-.015),'R6':(.126,.015)}"

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def replace_once(source,old,new):
    if source.count(old)!=1:raise ValueError('Unexpected frozen source: '+old[:90])
    return source.replace(old,new)

def pair_indices(cursor,n,count,device):
    import torch
    if min(n,count)<1 or cursor<0:raise ValueError('Invalid pair allocation')
    return torch.arange(cursor,cursor+n,device=device)%count,cursor+n

def freeze(parent,output,load_guard=False,upright_support=False):
    manifest=json.loads((parent.parent/'manifest.json').read_text())
    if sha(parent)!=manifest['policy_sha256']:raise ValueError('Parent weights changed')
    for name,digest in manifest['source_hashes'].items():
        if sha(parent.parent/'sources'/name)!=digest:raise ValueError('Parent source changed')
    shutil.copytree(parent.parent,output,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    path=output/'sources/autonomy_env.py';source=path.read_text()
    source=replace_once(source,"SOURCES={'L5':(0.,0.),'L6':(0.,-.030)}",POSTS_SOURCE)
    source=replace_once(source,"TARGETS={'R1':(.042,-.015),'R2':(.042,.015)}",POSTS_TARGET)
    source=replace_once(source,'        self.support=None','        self.support=None\n        self.paired_posts=False;self.pair_cursor=0')
    marker='        target_idx=torch.randint(len(self.target_values),(n,),generator=self.reset_rng,device=dev)'
    source=replace_once(source,marker,marker+"\n        if self.paired_posts:\n            from autonomy_three_posts import pair_indices\n            source_idx,self.pair_cursor=pair_indices(self.pair_cursor,n,len(self.source_names),dev)\n            target_idx=source_idx")
    marker='source_post_center_m=self.source[i].cpu().tolist(),'
    source=replace_once(source,marker,marker+'target_post_center_m=self.target[i].cpu().tolist(),')
    path.write_text(source)
    path=output/'sources/autonomy_main.py';source=path.read_text()
    marker="    p.add_argument('--sources',nargs='+',default=['L5']);p.add_argument('--targets',nargs='+',default=['R2'])"
    source=replace_once(source,marker,marker+"\n    p.add_argument('--paired_posts',action='store_true',help='Reset-only round-robin pairing: sources[i] to targets[i]')")
    marker="    if args.video and (args.num_envs!=1 or args.episodes!=1):p.error('Video needs one environment and episode')"
    source=replace_once(source,marker,"    if args.video and (args.num_envs!=1 or args.episodes not in (1,3)):p.error('Video needs one environment and one or three episodes')\n    if args.paired_posts and len(args.sources)!=len(args.targets):p.error('Equal source/target lengths required')")
    marker='        vec.balance_direction=args.balance_direction'
    source=replace_once(source,marker,'        vec.paired_posts=args.paired_posts\n'+marker)
    marker='reset_only=True),'
    source=replace_once(source,marker,"reset_only=True,paired_posts=args.paired_posts,pair_order=list(zip(args.sources,args.targets)) if args.paired_posts else None),")
    if load_guard:
        marker="    p.add_argument('--paired_posts',action='store_true',help='Reset-only round-robin pairing: sources[i] to targets[i]')"
        source=replace_once(source,marker,marker+"\n    p.add_argument('--gripper_load_guard',action='store_true',help='Explicit temporary force relief on jaws; neural task decisions unchanged')")
        marker='        chunks=[];episode_id=torch.arange(args.num_envs,device=args.device);next_id=args.num_envs'
        source=replace_once(source,marker,"        from autonomy_gripper_guard import GripperLoadGuard\n        load_guard=GripperLoadGuard(args.num_envs,args.device) if args.gripper_load_guard else None\n"+marker)
        marker='                else:executed=net(x)'
        source=replace_once(source,marker,marker+'\n                if load_guard is not None:executed=load_guard.apply(executed,vec.state,vec.reference)')
        marker="        if any(row['assisted_grasp']!=report['assisted_grasp'] for row in rows):raise ValueError('Inconsistent assistance evidence')"
        source=replace_once(source,marker,"        if load_guard is not None:report['gripper_load_guard']=load_guard.evidence()\n"+marker)
        shutil.copy2(Path(__file__).parent/'autonomy_gripper_guard.py',output/'sources/autonomy_gripper_guard.py')
    if upright_support:
        if not load_guard:raise ValueError('This experiment requires the explicit load guard')
        marker="    p.add_argument('--gripper_load_guard',action='store_true',help='Explicit temporary force relief on jaws; neural task decisions unchanged')"
        source=replace_once(source,marker,marker+"\n    p.add_argument('--upright_grasp_support',action='store_true',help='Explicit bounded roll/pitch assistance after measured grasp')")
        marker='        vec.balance_direction=args.balance_direction'
        source=replace_once(source,marker,"        upright_evidence=None\n        if args.upright_grasp_support:\n            from autonomy_upright_support import install\n            upright_evidence=install(vec)\n"+marker)
        marker="        if load_guard is not None:report['gripper_load_guard']=load_guard.evidence()"
        source=replace_once(source,marker,marker+"\n        if upright_evidence is not None:report['upright_grasp_support']=upright_evidence()")
        support=output/'sources/autonomy_support.py'
        support.write_text(replace_once(support.read_text(),"(state['bottom']>.012)","(state['bottom']>.009)"))
        shutil.copy2(Path(__file__).parent/'autonomy_upright_support.py',output/'sources/autonomy_upright_support.py')
    path.write_text(source)
    shutil.copy2(__file__,output/'sources/autonomy_three_posts.py')
    changed={name for name,digest in manifest['source_hashes'].items() if sha(output/'sources'/name)!=digest}
    expected={'autonomy_env.py','autonomy_main.py'}|({'autonomy_support.py'} if upright_support else set())
    if changed!=expected or sha(output/'policy.pt')!=sha(parent):
        raise ValueError('Unexpected motor or weight change')
    extras=['autonomy_three_posts.py']+(['autonomy_gripper_guard.py'] if load_guard else [])+(['autonomy_upright_support.py'] if upright_support else [])
    for name in (*changed,*extras):manifest['source_hashes'][name]=sha(output/'sources'/name)
    manifest.update(parent_policy_sha256=sha(parent),implementation_change='Expanded physical post names; explicit paired resets; multi-episode front video; original neural actor and physics unchanged')
    manifest['temporary_gripper_load_guard_available']=load_guard
    manifest['temporary_upright_support_available']=upright_support
    manifest['original_native_plant_unchanged']=True
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return output/'policy.pt'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--load_guard',action='store_true')
    p.add_argument('--upright_support',action='store_true')
    a=p.parse_args()
    from preserve_autonomy import verify_archive
    archive=verify_archive(ROOT/'milestones/checkpoint11-assisted-full-bc-yaw5.json')
    print(freeze(ROOT/archive['preserved_policy'],a.output,a.load_guard,a.upright_support))

if __name__=='__main__':main()
