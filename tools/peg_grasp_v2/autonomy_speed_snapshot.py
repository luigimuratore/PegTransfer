"""Freeze checkpoint12 with extended travel caps; never edit the old archive."""
import argparse
import json
from pathlib import Path
import shutil
from autonomy_three_posts import sha
from preserve_autonomy import verify_archive

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def freeze(destination):
    milestone = ROOT/'milestones/checkpoint12-three-posts-assisted.json'
    old = verify_archive(milestone)
    parent = (ROOT/old['preserved_policy']).parent
    shutil.copytree(parent, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    folder = destination/'sources'
    env = folder/'autonomy_env.py'
    text = env.read_text()
    marker = "        else:raise ValueError('Unknown motion profile')"
    assert text.count(marker) == 1
    text = text.replace(marker,
        "        elif motion_profile=='fast_travel':\n"
        "            from autonomy_fast_transit import FastTransitMotorReference\n"
        "            self.reference=FastTransitMotorReference(self.state,profile,free_scale,transport_scale)\n"+marker)
    env.write_text(text)
    main = folder/'autonomy_main.py'
    text = main.read_text()
    replacements = {
        "choices=('contact','clearance','travel')": "choices=('contact','clearance','travel','fast_travel')",
        "Clearance profile only: stable isolated grasp transport multiplier, 1..4; interaction stays slow":
            "Stable isolated transport: 1..4, or 1..32 with fast_travel; interaction stays slow",
        "free_max=4 if args.motion_profile!='contact' else 2":
            "free_max=12 if args.motion_profile=='fast_travel' else (4 if args.motion_profile!='contact' else 2)",
        "    if not np.isfinite(args.transport_speed_scale) or not 1<=args.transport_speed_scale<=4:\n        p.error('Transport speed scale must be finite and within 1..4')":
            "    transport_max=32 if args.motion_profile=='fast_travel' else 4\n"
            "    if not np.isfinite(args.transport_speed_scale) or not 1<=args.transport_speed_scale<=transport_max:\n"
            "        p.error(f'Transport speed scale must be finite and within 1..{transport_max}')",
    }
    for before, after in replacements.items():
        assert text.count(before) == 1, before
        text = text.replace(before, after)
    main.write_text(text)
    shutil.copy2(HERE/'autonomy_fast_transit.py', folder/'autonomy_fast_transit.py')
    manifest = json.loads((destination/'manifest.json').read_text())
    for file in ('autonomy_env.py','autonomy_main.py','autonomy_fast_transit.py'):
        manifest['source_hashes'][file] = sha(folder/file)
    manifest.update(parent_checkpoint=str(parent/'policy.pt'),parent_policy_sha256=sha(parent/'policy.pt'),
        implementation_change='Extend measured stable transit caps and clear unloaded rotation; neural weights, near/contact caps, jaw control, task decisions and physics unchanged')
    assert manifest['policy_sha256'] == sha(destination/'policy.pt') == sha(parent/'policy.pt')
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return destination/'policy.pt'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    print(freeze(p.parse_args().output))
