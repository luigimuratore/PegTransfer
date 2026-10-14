"""Export verified complete teacher episodes for full-sequence Behavior Cloning."""
import argparse
import json
from pathlib import Path
from full_learning import export

REPO=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,default=REPO/'logs/grasp_v2/full_assisted_validation3_2026-10-05.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base_manifest',type=Path,default=REPO/'milestones/checkpoint3-full-assisted.json')
    args=p.parse_args()
    meta=export(args.report,args.output,args.base_manifest)
    print(json.dumps(dict(output=str(args.output),episodes=meta['episodes'],transitions=meta['transitions'],
        stages=len(meta['stages']),assisted_grasp=meta['assisted_grasp']),indent=2))

if __name__=='__main__':main()
