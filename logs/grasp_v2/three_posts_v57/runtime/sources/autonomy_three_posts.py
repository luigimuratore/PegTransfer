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

def freeze(parent,output):
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
    path.write_text(source)
    shutil.copy2(__file__,output/'sources/autonomy_three_posts.py')
    changed={name for name,digest in manifest['source_hashes'].items() if sha(output/'sources'/name)!=digest}
    if changed!={'autonomy_env.py','autonomy_main.py'} or sha(output/'policy.pt')!=sha(parent):
        raise ValueError('Unexpected motor or weight change')
    for name in (*changed,'autonomy_three_posts.py'):manifest['source_hashes'][name]=sha(output/'sources'/name)
    manifest.update(parent_policy_sha256=sha(parent),implementation_change='Expanded physical post names; explicit paired resets; multi-episode front video; original neural actor and physics unchanged')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return output/'policy.pt'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from preserve_autonomy import verify_archive
    archive=verify_archive(ROOT/'milestones/checkpoint11-assisted-full-bc-yaw5.json')
    print(freeze(ROOT/archive['preserved_policy'],a.output))

if __name__=='__main__':main()
