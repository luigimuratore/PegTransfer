"""Test a pressure-recovery repair on the rejected weights before retraining."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path.cwd().resolve()
HERE = Path(__file__).resolve().parent
FILES = ('residual_main.py', 'residual_env.py', 'residual_learning.py',
         'residual_grasp_guard.py', 'residual_contact_diagnostic.py',
         'residual_collision_support.py', 'residual_spring_support.py',
         'residual_pipeline.py', 'residual_recovery_retry.py')


def fork_candidate(checkpoint, destination, sources):
    from residual_pipeline import sha
    checkpoint, destination, sources = map(Path, (checkpoint, destination, sources))
    parent = checkpoint.parent
    manifest = json.loads((parent / 'manifest.json').read_text())
    if sha(checkpoint) != manifest['sac_sha256']:
        raise ValueError('Parent weights changed')
    for folder, key in (('base', 'base_hashes'), ('sources', 'source_hashes')):
        for name, digest in manifest[key].items():
            if sha(parent / folder / name) != digest:
                raise ValueError(f'Parent artifact changed: {folder}/{name}')
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(parent / 'base', destination / 'base', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (destination / 'sources').mkdir()
    for name in manifest['source_hashes']:
        shutil.copy2(sources / name, destination / 'sources' / name)
    shutil.copy2(checkpoint, destination / 'candidate.zip')
    manifest.update(recovery_before_lift=True, promoted=False,
                    parent_candidate=str(checkpoint.resolve()),
                    parent_manifest_sha256=sha(parent / 'manifest.json'),
                    inference_configuration_changed=True, additional_training=False,
                    physical_evaluation_required=True,
                    source_hashes={p.name: sha(p) for p in (destination / 'sources').iterdir()})
    manifest['training_parameters'] = dict(manifest['training_parameters'],
                                         recovery_before_lift=True)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return destination / 'candidate.zip'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--training_output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--transitions', type=int, default=12000)
    p.add_argument('--snapshot_ready', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    out = a.output.resolve()
    if not a.snapshot_ready:
        out.mkdir(parents=True, exist_ok=False)
        (out / 'workflow_sources').mkdir()
        for name in FILES: shutil.copy2(HERE / name, out / 'workflow_sources' / name)
        os.execv(sys.executable, [sys.executable, '-u', str(out / 'workflow_sources' / Path(__file__).name),
                                 *sys.argv[1:], '--snapshot_ready'])
    from residual_pipeline import sha, write, validate_child_result, comparison
    state = dict(complete=False, stage='repair_diagnostic',
                 original_checkpoint_preserved='checkpoint13', new_checkpoint_promoted=False,
                 workflow_source_hashes={n: sha(HERE / n) for n in FILES},
                 training_output=str(a.training_output.resolve()), results=[])
    write(out / 'status.json', state)

    def run(label, mode, candidate, extras):
        state['stage'] = label; write(out / 'status.json', state)
        command = [sys.executable, '-u', str(HERE / 'residual_main.py'), mode,
                   '--checkpoint', str(candidate), '--headless', '--output', str(out / label), *extras]
        write(out / f'{label}_command.json', command)
        with (out / f'{label}.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        validate_child_result(out / label, mode, video='--video' in extras)
        report_path = out / label / 'evaluation.json'
        result = json.loads(report_path.read_text())
        state['results'].append(dict(label=label, report=str(report_path), sha256=sha(report_path),
                                     success=result['success'], episodes=result['episodes']))
        write(out / 'status.json', state)
        return result

    try:
        repaired = fork_candidate(a.checkpoint.resolve(), out / 'repaired_candidate', HERE)
        baseline = json.loads(a.baseline.read_text())
        if baseline['success'] != 12 or not baseline['complete'] or baseline['seed'] != 59002:
            raise ValueError('Expected the complete paired V4 baseline12')
        result = run('repair12', 'evaluate', repaired,
                     ['--num_envs', '12', '--episodes', '12', '--seed', '59002',
                      '--jitter_mm', '.1', '--yaw_deg', '.5'])
        # Even a rejected experimental bundle gets its own complete-episode video.
        video = run('repair_video', 'play', repaired,
                    ['--video', '--num_envs', '1', '--episodes', '1', '--seed', '59003'])
        video_path = ROOT / 'logs/grasp_v2/videos' / f'{out.name}.mp4'
        for source, target in ((out/'repair_video/inference_front.mp4', video_path),
                               (out/'repair_video/video.json', video_path.with_suffix('.json')),
                               (out/'repair_video/evaluation.json', video_path.with_suffix('.evaluation.json'))):
            with target.open('xb') as stream: stream.write(source.read_bytes())
        state['video'] = str(video_path)
        state['repair_passed'] = comparison(baseline, result) and video['success'] == 1
        if not state['repair_passed']:
            state.update(complete=True, stage='repair_gate_failed')
            write(out/'status.json', state)
            return 1
        state['stage'] = 'retraining_pipeline'; write(out/'status.json', state)
        command = [sys.executable, '-u', str(HERE/'residual_pipeline.py'),
                   '--output', str(a.training_output.resolve()), '--transitions', str(a.transitions),
                   '--held_jaw_guard', '--grasp_stiffness', '900', '--spring_scope', 'donor_recovery',
                   '--held_jaw_recovery_only', '--recovery_before_lift']
        write(out/'training_command.json', command)
        with (out/'training_pipeline.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        trained = json.loads((a.training_output/'status.json').read_text())
        if not trained.get('complete') or trained.get('stage') != 'pilot_passed':
            state.update(complete=True, stage='training_gate_failed', training_result=trained)
            write(out/'status.json', state)
            return 1
        independent = run('independent12', 'evaluate', a.training_output/'training/candidate.zip',
                          ['--num_envs', '12', '--episodes', '12', '--seed', '62002',
                           '--jitter_mm', '.1', '--yaw_deg', '.5'])
        state.update(complete=True, stage='confirmed_candidate' if independent['success']==12 else 'independent_gate_failed',
                     candidate=str(a.training_output/'training/candidate.zip'),
                     assisted_grasp=True, native_full_policy_verified=False)
        write(out/'status.json', state)
        return 0 if independent['success']==12 else 1
    except Exception as error:
        state.update(complete=True, stage='stopped_on_error', error=repr(error))
        write(out/'status.json', state)
        raise


if __name__ == '__main__': raise SystemExit(main())
