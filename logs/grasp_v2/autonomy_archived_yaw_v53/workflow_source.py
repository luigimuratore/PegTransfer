"""Test +/-5-degree resets with checkpoint10's exact archived implementation.

Only the reset yaw changes. No repackaging, training, changed assistance,
changed contact thresholds or new permanent checkpoint is implied.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from autonomy_motion_curriculum import distribution_gate, sha
from preserve_autonomy import verify_archive

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    milestone_path = ROOT/'milestones/checkpoint10-assisted-full-bc-transit.json'
    milestone = verify_archive(milestone_path)
    policy = ROOT/milestone['preserved_policy']
    # Preserve all attempts, including V51's lost baseline case; V52 does
    # not overwrite it or prove a cause for the difference between runs.
    audit_path = ROOT/'logs/grasp_v2/autonomy_baseline_audit_v52/status.json'
    audit = json.loads(audit_path.read_text())
    report_path = Path(audit['report'])
    closure = json.loads((report_path.parent/'environment_close.json').read_text())
    if (audit.get('complete') is not True or audit.get('archived_success') != 16
        or audit.get('policy_sha256') != sha(policy)
        or sha(report_path) != audit['report_sha256']
        or closure.get('environment_closed') is not True
        or closure.get('evaluation_sha256') != sha(report_path)):
        raise ValueError('Verified V52 archived baseline required')
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output/'workflow_source.py')
    state = dict(complete=False, stage='starting', training=False,
        algorithm='Frozen checkpoint10 BC generalization screening; not new training',
        policy=str(policy), policy_sha256=sha(policy), milestone_sha256=sha(milestone_path),
        frozen_implementation=True, results=[], yaw_deg=5., jitter_mm=.5,
        sources=['L5'], targets=['R2'], assisted_grasp=True, native_full_policy_verified=False,
        gate='At least15/16 complete physically measured tasks on each seed2468 and54321',
        baseline_evidence=str(audit_path), v51_failure_retained=True)
    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = args.output/'status.json.tmp'
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(args.output/'status.json')
    try:
        save()
        for seed in (2468, 54321):
            label = f'yaw5_seed{seed}'
            out = args.output/label
            state['stage'] = label
            save()
            print(f'[ARCHIVED YAW] {label}', flush=True)
            with (args.output/f'{label}.log').open('w') as log:
                child = subprocess.Popen([sys.executable, '-u', str(HERE/'autonomy_replay.py'),
                    'evaluate', '--policy', str(policy), '--milestone', str(milestone_path),
                    '--output', str(out), '--skill', 'full', '--num_envs', '16', '--episodes', '16',
                    '--episode_steps', '10000', '--seed', str(seed), '--sources', 'L5', '--targets', 'R2',
                    '--jitter_mm', '.5', '--yaw_deg', '5', '--grasp_support', '1', '--support_profile', 'firm',
                    '--placement_support', '--motion_profile', 'travel', '--free_speed_scale', '2',
                    '--transport_speed_scale', '2', '--headless', '--diagnostic_events', '--diagnostic_interval', '25'],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid
                save()
                try:
                    code = child.wait()
                except BaseException:
                    child.send_signal(signal.SIGINT)
                    child.wait()
                    raise
            state.pop('child_pid', None)
            if code not in (0, 1):
                raise RuntimeError(f'{label} exited {code}')
            path = out/'evaluation.json'
            report = json.loads(path.read_text())
            close = json.loads((out/'environment_close.json').read_text())
            if (close.get('environment_closed') is not True or close.get('evaluation_complete') is not True
                or close.get('evaluation_sha256') != sha(path) or report.get('seed') != seed
                or report.get('policy_sha256') != sha(policy)):
                raise ValueError('Incomplete or incompatible physical evidence')
            speed = report.get('motion_reference', {})
            if (speed.get('profile') != 'travel' or speed.get('free_scale') != 2.
                or speed.get('transport_scale') != 2. or speed.get('contact_scale') != 1.):
                raise ValueError('Archived speed configuration changed')
            passed = distribution_gate(report, .5, 5., 16, 15/16)
            if passed and code:
                raise ValueError('Successful gate exited unsuccessfully')
            state['results'].append(dict(seed=seed, success=report['success'], episodes=16, passed=passed,
                report=str(path), report_sha256=sha(path),
                failed_cases=[dict(episode_id=i, option=c['terminal_learned_option'],
                    grasp_psm1=c['grasp_psm1'], lift_stable=c['lift_stable'], grasp_psm2=c['grasp_psm2'],
                    handover=c['handover'], peak_force_n=c['peak_force_n'], timeout=c['timeout'], failure=c['failure'])
                    for i,c in enumerate(report['cases']) if not c['success']]))
            save()
            print(f"[ARCHIVED YAW] {label}: {report['success']}/16 passed={passed}", flush=True)
        accepted = all(row['passed'] for row in state['results'])
        verify_archive(milestone_path)
        state.update(complete=True, stage='yaw5_verified' if accepted else 'yaw5_recovery_required',
            physical_success=sum(row['success'] for row in state['results']), episodes=32,
            training_required=not accepted,
            next_step='Extend the reset range in an isolated runtime and recheck baseline before testing10 degrees' if accepted
                else 'Use the recorded failures for targeted supervised recovery; do not enlarge yaw yet')
        save()
        return 0 if accepted else 1
    except BaseException as error:
        state.update(complete=False, stage='interrupted' if isinstance(error, KeyboardInterrupt) else 'workflow_error',
                     error=f'{type(error).__name__}: {error}')
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
