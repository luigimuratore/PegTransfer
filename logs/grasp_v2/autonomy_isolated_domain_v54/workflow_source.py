"""Expand resets while copying checkpoint10's original runtime byte for byte.

The only source change permitted is the CLI range check, to allow 1.1mm/30deg.
Control, physics, observations, neural parameters and reset sampling stay old.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from autonomy_motion_curriculum import sha, distribution_gate
from geometry import load_profile, validate_reset_range
from preserve_autonomy import verify_archive

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SEEDS = (2468, 54321)


def extend_range_check(source):
    old = "    if args.jitter_mm<0 or args.jitter_mm>1 or not 0<=args.yaw_deg<=5:p.error('First curriculum: jitter 0..1 mm and yaw 0..5 deg')"
    new = "    if not np.isfinite(args.jitter_mm) or args.jitter_mm<0 or args.jitter_mm>1.1 or not 0<=args.yaw_deg<=30:p.error('Isolated reset curriculum: jitter 0..1.1 mm and yaw 0..30 deg')"
    if source.count(old) != 1:
        raise ValueError('Archived range check differs; do not alter unknown runtime')
    return source.replace(old, new)


def freeze_runtime(parent, output):
    """Copy every preserved module; modify no motor/actor/reset implementation."""
    original = json.loads((parent.parent/'manifest.json').read_text())
    if sha(parent) != original['policy_sha256']:
        raise ValueError('Parent weights differ from their manifest')
    for name,digest in original['source_hashes'].items():
        if sha(parent.parent/'sources'/name) != digest:
            raise ValueError('Preserved parent source changed')
    # Check the entire admitted square against the mesh before widening its cap.
    clearance = validate_reset_range(load_profile(parent.parent/'sources/peg_profile.json'), 1.1, 30.)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(parent.parent/'sources', output/'sources', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(parent, output/'policy.pt')
    main = output/'sources/autonomy_main.py'
    source = main.read_text()
    main.write_text(extend_range_check(source))
    changed = [name for name,digest in original['source_hashes'].items() if sha(output/'sources'/name) != digest]
    if changed != ['autonomy_main.py'] or sha(output/'policy.pt') != sha(parent):
        raise ValueError('Isolated runtime changed more than its argument range check')
    manifest = dict(original)
    manifest.update(source_hashes={name:sha(output/'sources'/name) for name in original['source_hashes']},
                    parent_checkpoint=str(parent.resolve()), parent_policy_sha256=sha(parent),
                    implementation_change='Only the CLI allowed reset range; original frozen control/reset/actor/physics sources unchanged',
                    retrained=False)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    training = json.loads((parent.parent/'training.json').read_text())
    training.update(retrained=False, neural_parameters_unchanged=True, parent_policy=str(parent.resolve()),
                    physical_evaluation_required=True, implementation_change=manifest['implementation_change'])
    (output/'training.json').write_text(json.dumps(training, indent=2)+'\n')
    evidence = dict(policy_sha256=sha(parent), changed_source_files=changed,
                    unchanged_source_files=len(original['source_hashes'])-1,
                    reset_square_minimum_post_clearance_mm=clearance*1000,
                    allowed_yaw_deg=30., allowed_jitter_half_width_mm=1.1,
                    controller_and_sampling_unchanged=True, weights_unchanged=True)
    (output/'range_extension.json').write_text(json.dumps(evidence, indent=2)+'\n')
    return evidence


def retained_successes(reference, candidate):
    old, new = reference.get('cases', []), candidate.get('cases', [])
    if (reference.get('policy_sha256') != candidate.get('policy_sha256')
        or reference.get('seed') != candidate.get('seed') or not old or len(old) != len(new)):
        return False
    for a,b in zip(old, new):
        if any(a.get('reset', {}).get(k) != b.get('reset', {}).get(k)
               for k in ('source', 'target', 'jitter_mm', 'yaw_deg')):
            return False
        if a.get('success') and not b.get('success'):
            return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    milestone = verify_archive(ROOT/'milestones/checkpoint11-assisted-full-bc-yaw5.json')
    parent = ROOT/milestone['preserved_policy']
    prior_path = ROOT/'logs/grasp_v2/autonomy_archived_yaw_v53/status.json'
    prior = json.loads(prior_path.read_text())
    if prior.get('complete') is not True or prior.get('stage') != 'yaw5_verified' or prior.get('policy_sha256') != sha(parent):
        raise ValueError('Completed V53 two-seed yaw5 validation required')
    references = {}
    for row in prior['results']:
        path = Path(row['report']);report = json.loads(path.read_text())
        close = json.loads((path.parent/'environment_close.json').read_text())
        if (sha(path) != row['report_sha256'] or close.get('evaluation_sha256') != sha(path)
            or close.get('environment_closed') is not True
            or not distribution_gate(report,.5,5.,16,15/16)):
            raise ValueError('Unverified V53 reference')
        references[report['seed']] = report
    if set(references) != set(SEEDS):
        raise ValueError('Missing V53 seed')
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output/'workflow_source.py')
    levels = [dict(label='runtime_yaw5', jitter=.5, yaw=5.),
              *[dict(label=f'yaw{yaw:g}', jitter=.5, yaw=yaw) for yaw in (10.,20.,30.)],
              *[dict(label=f'offset{jitter:g}', jitter=jitter, yaw=30.) for jitter in (.75,1.,1.1)]]
    state = dict(complete=False, stage='freeze_runtime', training=False, results=[], verified_levels=[],
                 policy_sha256=sha(parent), parent_checkpoint='checkpoint11', levels=levels,
                 assisted_grasp=True, native_full_policy_verified=False,
                 gate='At least15/16 per seed; yaw5 runtime comparison also retains every V53 success',
                 sources=['L5'], targets=['R2'])
    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = args.output/'status.json.tmp';temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(args.output/'status.json')
    try:
        save()
        runtime = args.output/'runtime'
        state['range_extension'] = freeze_runtime(parent, runtime)
        policy = runtime/'policy.pt'
        state['runtime_policy'] = str(policy)
        save()
        for level in levels:
            accepted = True
            for seed in SEEDS:
                label = f"{level['label']}_seed{seed}"
                state['stage'] = label
                save()
                out = args.output/label
                print(f'[ISOLATED DOMAIN] {label}', flush=True)
                with (args.output/f'{label}.log').open('w') as log:
                    child = subprocess.Popen([sys.executable,'-u',str(HERE/'autonomy_replay.py'),'evaluate',
                        '--policy',str(policy),'--output',str(out),'--skill','full','--num_envs','16','--episodes','16',
                        '--episode_steps','10000','--seed',str(seed),'--sources','L5','--targets','R2',
                        '--jitter_mm',str(level['jitter']),'--yaw_deg',str(level['yaw']),
                        '--grasp_support','1','--support_profile','firm','--placement_support',
                        '--motion_profile','travel','--free_speed_scale','2','--transport_speed_scale','2',
                        '--headless','--diagnostic_events','--diagnostic_interval','25'],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    state['child_pid'] = child.pid
                    save()
                    try:code = child.wait()
                    except BaseException:
                        child.send_signal(signal.SIGINT);child.wait();raise
                state.pop('child_pid',None)
                if code not in (0,1):raise RuntimeError(f'{label} exited {code}')
                path = out/'evaluation.json';report = json.loads(path.read_text())
                close = json.loads((out/'environment_close.json').read_text())
                if (close.get('environment_closed') is not True or close.get('evaluation_complete') is not True
                    or close.get('evaluation_sha256') != sha(path) or report.get('policy_sha256') != sha(policy)
                    or report.get('seed') != seed):raise ValueError('Physical report/closure/weights mismatch')
                speed = report.get('motion_reference',{})
                if (speed.get('profile') != 'travel' or speed.get('free_scale') != 2
                    or speed.get('transport_scale') != 2 or speed.get('contact_scale') != 1):
                    raise ValueError('Motion settings changed')
                passed = distribution_gate(report,level['jitter'],level['yaw'],16,15/16)
                retained = retained_successes(references[seed],report) if level['label'] == 'runtime_yaw5' else None
                if retained is not None:passed &= retained
                if passed and code:raise ValueError('Successful gate exited unsuccessfully')
                accepted &= passed
                state['results'].append(dict(label=label,seed=seed,jitter_mm=level['jitter'],yaw_deg=level['yaw'],
                    success=report['success'],episodes=16,passed=passed,baseline_successes_retained=retained,
                    report=str(path),report_sha256=sha(path),failed_episode_ids=[i for i,c in enumerate(report['cases']) if not c['success']]))
                save()
                print(f"[ISOLATED DOMAIN] {label}: {report['success']}/16 passed={passed}",flush=True)
            if not accepted:
                state.update(complete=True,stage='domain_gate_failed',failed_level=level,training_required=True,
                             reason='Do not widen resets further; retain failed cases and the previous verified condition')
                save();return 1
            state['verified_levels'].append(level)
            save()
        state.update(complete=True,stage='same_post_range_verified',training_required=False,
                     next_step='Introduce new post pairs with small offset/yaw; reachability and physical completion require separate validation')
        save();return 0
    except BaseException as error:
        state.update(complete=False,stage='interrupted' if isinstance(error,KeyboardInterrupt) else 'workflow_error',
                     error=f'{type(error).__name__}: {error}')
        save();raise


if __name__ == '__main__':
    raise SystemExit(main())
