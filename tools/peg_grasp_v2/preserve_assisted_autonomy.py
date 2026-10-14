"""Preserve an explicitly assisted learned full pilot without native certification."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from autonomy_assistance_curriculum import assisted_gate
from preserve_autonomy import ROOT, sha, verify_archive


def preserve(policy, pilot_report, destination, milestone):
    policy, pilot_report, destination, milestone = [Path(p).resolve()
        for p in (policy, pilot_report, destination, milestone)]
    if destination.exists() or milestone.exists():
        raise ValueError('Preserved checkpoints must not be overwritten')
    # Check all provenance before copying anything into the archive.
    digest = sha(policy)
    report = json.loads(pilot_report.read_text())
    strength = report.get('grasp_support', {}).get('strength', 0.)
    config = report.get('grasp_support', {})
    if config.get('profile'):
        from autonomy_constraints_gate import assisted_gate as constraints_gate
        passed = constraints_gate(report, strength, 3, 1., config['profile'], config['placement_guide_enabled'])
        close_path = pilot_report.parent / 'environment_close.json'
        closed = json.loads(close_path.read_text())
        if (closed.get('environment_closed') is not True or closed.get('evaluation_complete') is not True
                or closed.get('evaluation_sha256') != sha(pilot_report)):
            raise ValueError('Constraint pilot must close its environment successfully')
    else:
        passed = assisted_gate(report, strength, 3, 1.)
        close_path = None
    if not 0 < strength <= 1 or report.get('policy_sha256') != digest or not passed:
        raise ValueError('Same-weight completed 3/3 assisted full pilot with released support required')
    manifest = json.loads((policy.parent / 'manifest.json').read_text())
    if manifest['policy_sha256'] != digest:
        raise ValueError('Model weights changed')
    for name, expected in manifest['source_hashes'].items():
        if sha(policy.parent / 'sources' / name) != expected:
            raise ValueError(f'Model source changed: {name}')
    physics = ROOT / 'milestones/checkpoint3-full-assisted.json'
    if sha(physics) != manifest['physics_manifest_sha256']:
        raise ValueError('Original physics changed')
    destination.relative_to(ROOT)
    milestone.relative_to(ROOT)
    destination.mkdir(parents=True)
    shutil.copytree(policy.parent, destination / 'model', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    reports = destination / 'reports'
    reports.mkdir()
    shutil.copy2(pilot_report, reports / 'assisted_pilot.json')
    shutil.copy2(__file__, reports / 'preservation_source.py')
    shutil.copy2(Path(__file__).with_name('autonomy_assistance_curriculum.py'), reports / 'assisted_gate_source.py')
    if close_path:
        shutil.copy2(close_path, reports / 'environment_close.json')
        shutil.copy2(Path(__file__).with_name('autonomy_constraints_gate.py'), reports / 'constraints_gate_source.py')
    result = dict(name=destination.name, date=datetime.now(timezone.utc).date().isoformat(),
        algorithm='Structured hierarchical Behavior Cloning; not SAC',
        preserved_bundle=str(destination.relative_to(ROOT)),
        preserved_policy=str((destination / 'model/policy.pt').relative_to(ROOT)),
        policy_sha256=digest, physics_manifest=str(physics.relative_to(ROOT)),
        physics_manifest_sha256=sha(physics), assisted_full_pilot_verified=True,
        native_full_policy_verified=False, assisted_grasp=True,
        teacher_at_inference=False, programmed_task_controller=False,
        pilot_success=3, pilot_episodes=3, seed=report['seed'],
        grasp_support=report['grasp_support'],
        replay_settings=dict(skill='full', grasp_support=strength, num_envs=1, episodes=1,
            episode_steps=10000, sources=['L5'], targets=['R2'], jitter_mm=0, yaw_deg=0, seed=123),
        limitations=['Three fixed L5 to R2 pilot episodes; randomized robustness not certified',
            'A virtual spring assists the grasp; not a native full checkpoint',
            'Simulator state inputs, fitted pose decoders and generic IK; not vision or SAC',
            'Rigid peg with compliant contacts; not calibrated deformable rubber',
            'Original Isaac/Conda environment and preserved checkpoint3 physics assets required'],
        files={str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob('*')) if p.is_file()})
    if config.get('profile'):
        result['replay_settings'].update(support_profile=config['profile'], placement_support=config['placement_guide_enabled'])
    if config.get('placement_guide_enabled'):
        result['limitations'][1] = 'Virtual grasp spring and a scripted free-release centering guide; not native full control'
    milestone.parent.mkdir(parents=True, exist_ok=True)
    with milestone.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    verify_archive(milestone)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--pilot_report', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--milestone', type=Path, required=True)
    a = p.parse_args()
    result = preserve(a.policy, a.pilot_report, a.destination, a.milestone)
    print(f"Preserved {result['name']}: 3/3 assisted full pilot; native full NOT certified")


if __name__ == '__main__':
    main()
