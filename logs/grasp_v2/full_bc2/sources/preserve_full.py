"""Preserve a successful assisted teacher and its original scene dependencies."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

from audit_full import audit, sha256

REPO = Path(__file__).resolve().parents[2]


def preserve(reference, destination, manifest_path):
    checked = audit(reference)
    report = json.loads(reference.read_text())
    if destination.exists() or manifest_path.exists():
        raise ValueError('Preserved bundles and manifests must not be overwritten')
    for name, version in report['versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Environment differs from the reference: {name}')
    if checked['full_success'] < 3:
        raise ValueError('Preserve the full baseline after at least three successful episodes')
    destination.mkdir(parents=True)
    shutil.copytree(report['source_snapshot'], destination / 'tools/peg_grasp_v2')
    for extension in ('robotic.surgery.tasks', 'robotic.surgery.assets'):
        relative = Path('workflows/robotic_surgery/scripts/simulation/exts') / extension
        shutil.copytree(REPO / relative, destination / relative,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'))
    relative = Path('workflows/robotic_surgery/scripts/simulation/utils')
    shutil.copytree(REPO / relative, destination / relative,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    asset_root = Path('data/Isaac/Healthcare/0.5.0/132c82d')
    for asset in ('Props/PegBlock', 'Props/Table'):
        shutil.copytree(REPO / asset_root / asset, destination / asset_root / asset)
    profile = json.loads((destination / 'tools/peg_grasp_v2/peg_profile.json').read_text())
    robot = report['parameters'].get('robot_usd')
    candidates = [Path(robot)] if robot else [
        REPO / asset_root / 'Robots/dVRK/PSM/psm.usd',
        Path.home() / '.cache/ov/client/https/d36f3a4d2205f4288e3dfc1961a891b4254884432f41d7822f54437aca9b2d45.usd']
    source = next((path for path in candidates if path.is_file()), None)
    if source is None or sha256(source) != profile['robot']['asset_sha256']:
        raise ValueError('Original robot asset is missing or changed')
    target = destination / asset_root / 'Robots/dVRK/PSM/psm.usd'
    target.parent.mkdir(parents=True)
    shutil.copy2(source, target)
    model = Path(report['donor_policy'])
    shutil.copytree(model.parent, destination / 'donor_checkpoint',
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(reference, destination / 'reference.json')
    (destination / 'audit.json').write_text(json.dumps(checked, indent=2) + '\n')
    parameters = {key: value for key, value in report['parameters'].items() if key in (
        'stage_steps', 'seed', 'mass_g', 'friction', 'contact_stiffness', 'contact_damping',
        'jaw_torque', 'jaw_stiffness', 'grip_force', 'max_force', 'preload_seconds',
        'free_speed', 'contact_speed', 'lift_speed', 'rotation_speed', 'donor_depth_mm',
        'receiver_depth_mm', 'donor_open', 'receiver_open', 'peg_support',
        'peg_support_type', 'peg_support_window', 'camera_eye', 'camera_target',
        'donor_balance', 'donor_compliance')}
    manifest = dict(name='checkpoint3-full-assisted', date='2026-10-05',
        artifact_kind='programmed full-task teacher with saved BC donor pickup',
        source='L5', target='R2', full_task_learned_policy=False,
        assisted_grasp=report['assisted_grasp'], unassisted_full_success=0,
        full_success=report['full_success'], requested_episodes=report['requested_episodes'],
        reference_report=str(reference.relative_to(REPO)),
        reference_report_sha256=sha256(reference), bundle_root=str(destination.relative_to(REPO)),
        donor_checkpoint_sha256=report['donor_policy_sha256'],
        versions={**report['versions'], **{name: importlib.metadata.version(name)
            for name in ('numpy', 'scipy')}}, parameters=parameters,
        measurements=checked['measurements'],
        files_sha256={str(path.relative_to(destination)): sha256(path)
            for path in sorted(destination.rglob('*')) if path.is_file()},
        limitations=['Ideal unlimited fixed support from donor transfer through receiver insertion',
            'Support removed before receiver release and final free-placement verification',
            'No full-task neural policy trained; grasp choice and remaining motion are programmed',
            'Fixed L5 to R2 scene; no domain randomization',
            'Rigid peg with compliant contacts, not a calibrated deformable body',
            'Requires original Isaac environment and robot asset dependency cache'])
    with manifest_path.open('x') as file:
        json.dump(manifest, file, indent=2); file.write('\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--destination', type=Path, default=REPO / '.checkpoint_runs/checkpoint3-full-assisted')
    parser.add_argument('--manifest', type=Path, default=REPO / 'milestones/checkpoint3-full-assisted.json')
    args = parser.parse_args()
    result = preserve(args.reference.resolve(), args.destination.resolve(), args.manifest.resolve())
    print(f"Preserved {result['full_success']}/{result['requested_episodes']} full transfers: {args.manifest}")


if __name__ == '__main__':
    main()
