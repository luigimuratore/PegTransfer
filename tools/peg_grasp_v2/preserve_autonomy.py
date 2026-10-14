"""Archive a native full BC policy only after its two-seed physical certification."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from autonomy_full_recovery import native_full_gate

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def certification_evidence(policy, certification):
    policy = Path(policy).resolve()
    certification = Path(certification).resolve()
    status = json.loads((certification / 'status.json').read_text())
    digest = sha(policy)
    if (status.get('complete') is not True or status.get('native_full_policy_verified') is not True
            or status.get('stage') != 'certified' or status.get('policy_sha256') != digest):
        raise ValueError('Completed native full two-seed certification required')
    pilot = Path(status['pilot_report'])
    if not pilot.is_absolute():
        pilot = ROOT / pilot
    report = json.loads(pilot.read_text())
    if sha(pilot) != status['pilot_report_sha256'] or report.get('policy_sha256') != digest or not native_full_gate(report, 3, 1.):
        raise ValueError('Invalid or changed 3/3 pilot')
    evidence = [('pilot', pilot, report)]
    results = status.get('results', [])
    if len(results) != 2 or {row.get('seed') for row in results} != {123, 9876}:
        raise ValueError('Both independent evaluation seeds are required')
    for row in results:
        path = Path(row['report'])
        if not path.is_absolute():
            path = ROOT / path
        report = json.loads(path.read_text())
        if (row.get('passed') is not True or sha(path) != row['report_sha256']
                or report.get('policy_sha256') != digest or report.get('seed') != row['seed']
                or not native_full_gate(report, 16, .9)):
            raise ValueError('Invalid or changed two-seed evaluation')
        evidence.append((f"seed{row['seed']}", path, report))
    for _, _, report in evidence:
        if report.get('programmed_task_controller') is not False:
            raise ValueError('Programmed task control cannot certify a learned policy')
    return status, evidence


def verify_archive(milestone):
    manifest = json.loads(Path(milestone).read_text())
    folder = ROOT / manifest['preserved_bundle']
    for name, expected in manifest['files'].items():
        path = folder / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Preserved artifact missing or changed: {name}')
    if sha(ROOT / manifest['physics_manifest']) != manifest['physics_manifest_sha256']:
        raise ValueError('Original physics manifest changed')
    return manifest


def preserve(policy, certification, destination, milestone, video_report=None, video=None):
    status, evidence = certification_evidence(policy, certification)
    policy = Path(policy).resolve()
    destination = Path(destination).resolve()
    milestone = Path(milestone).resolve()
    if destination.exists() or milestone.exists():
        raise ValueError('Preserved checkpoints must not be overwritten')
    manifest = json.loads((policy.parent / 'manifest.json').read_text())
    if manifest['policy_sha256'] != sha(policy):
        raise ValueError('Model changed')
    for name, expected in manifest['source_hashes'].items():
        if sha(policy.parent / 'sources' / name) != expected:
            raise ValueError(f'Model source changed: {name}')
    physics = ROOT / 'milestones/checkpoint3-full-assisted.json'
    if sha(physics) != manifest['physics_manifest_sha256']:
        raise ValueError('Original physics changed')
    for name in ('certification_source.py', 'certification_gate_source.py'):
        if sha(Path(certification) / name) != status['certification_source_hashes'][name]:
            raise ValueError('Certification implementation changed')
    if bool(video_report) != bool(video):
        raise ValueError('Provide both video and its physical evaluation report')
    if video_report:
        replay = json.loads(Path(video_report).read_text())
        if replay.get('policy_sha256') != sha(policy) or not native_full_gate(replay, 1, 1.):
            raise ValueError('Video must show a completed native full success with the same weights')
    destination.mkdir(parents=True)
    shutil.copytree(policy.parent, destination / 'model', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    reports = destination / 'reports'
    reports.mkdir()
    for label, path, _ in evidence:
        shutil.copy2(path, reports / f'{label}.json')
    shutil.copy2(Path(certification) / 'status.json', reports / 'certification.json')
    for name in ('certification_source.py', 'certification_gate_source.py'):
        path = Path(certification) / name
        shutil.copy2(path, reports / name)
    if video:
        (destination / 'videos').mkdir()
        shutil.copy2(video, destination / 'videos/inference_front.mp4')
        shutil.copy2(video_report, reports / 'video_evaluation.json')
    result = dict(name=destination.name, date=datetime.now(timezone.utc).date().isoformat(),
        algorithm='Structured hierarchical Behavior Cloning: neural behavior/jaw decisions and fitted pose decoders; not SAC',
        policy=str(policy.relative_to(ROOT)), preserved_policy=str((destination / 'model/policy.pt').relative_to(ROOT)),
        preserved_bundle=str(destination.relative_to(ROOT)), policy_sha256=sha(policy),
        native_full_policy_verified=True, teacher_at_inference=False, assisted_grasp=False,
        programmed_task_controller=False, physics_manifest=str(physics.relative_to(ROOT)), physics_manifest_sha256=sha(physics),
        conditions=status['conditions'], confirmation_success=sum(r['success'] for _, _, r in evidence[1:]),
        confirmation_episodes=32, pilot_success=3,
        settings=dict(observation_dim=98, action_dim=14, learned_options=19, device='cuda:0',
            control_hz=100, physics_hz=400, free_speed_mm_s=25, contact_speed_mm_s=3,
            training_episodes=15, held_out_whole_episodes=4),
        limitations=['Verified only for L5 to R2 with small reset perturbations',
            'Structured pose decoders fitted from successful TRAIN demonstrations; not an unconstrained end-to-end action policy',
            'Simulator state inputs, not camera-based perception',
            'Rigid peg with compliant contacts; not calibrated deformable rubber or sim-to-real validation',
            'Original Isaac/Conda environment and preserved checkpoint3 physics assets required'],
        files={str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob('*')) if p.is_file()})
    milestone.parent.mkdir(parents=True, exist_ok=True)
    with milestone.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    verify_archive(milestone)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--certification', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--milestone', type=Path, required=True)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--video_report', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        result = verify_archive(args.milestone)
        print(f"Verified {result['name']}: {result['confirmation_success']}/32 native full, files unchanged")
        return
    if not all((args.policy, args.certification, args.destination)):
        parser.error('Specify policy, certification and destination')
    result = preserve(args.policy, args.certification, args.destination, args.milestone, args.video_report, args.video)
    print(f"Preserved {result['name']}: {result['confirmation_success']}/32 native full")


if __name__ == '__main__':
    main()
