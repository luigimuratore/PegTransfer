"""Replay the archived model to diagnose V51's lost baseline handover case."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
from autonomy_motion_curriculum import sha, distribution_gate
from preserve_autonomy import verify_archive

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    milestone = verify_archive(ROOT/'milestones/checkpoint10-assisted-full-bc-transit.json')
    policy = ROOT/milestone['preserved_policy']
    reference = ROOT/milestone['preserved_bundle']/'reports/transit0_seed54321.json'
    previous = ROOT/'logs/grasp_v2/autonomy_domain_v51/baseline_L5_R2_seed54321/evaluation.json'
    v51 = json.loads(previous.read_text())
    close = json.loads((previous.parent/'environment_close.json').read_text())
    if close.get('environment_closed') is not True or close.get('evaluation_sha256') != sha(previous):
        raise ValueError('V51 report is not verified')
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(complete=False, stage='archived_checkpoint10_diagnostic', training=False,
                 policy_sha256=sha(policy), checkpoint10_unchanged=True,
                 original_reference_sha256=sha(reference), v51_report_sha256=sha(previous))
    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        tmp = args.output/'status.json.tmp'
        tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(args.output/'status.json')
    def run(module, argv, label):
        state['stage'] = label
        save()
        with (args.output/f'{label}.log').open('w') as log:
            child = subprocess.Popen([sys.executable, '-u', str(HERE/module), *map(str, argv)],
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
        save()
        return code
    try:
        out = args.output/'archived_seed54321'
        code = run('autonomy_replay.py', ['evaluate', '--policy', policy,
            '--milestone', ROOT/'milestones/checkpoint10-assisted-full-bc-transit.json',
            '--output', out, '--skill', 'full', '--num_envs', 16, '--episodes', 16,
            '--episode_steps', 10000, '--seed', 54321, '--sources', 'L5', '--targets', 'R2',
            '--jitter_mm', .5, '--yaw_deg', 2, '--grasp_support', 1, '--support_profile', 'firm',
            '--placement_support', '--motion_profile', 'travel', '--free_speed_scale', 2,
            '--transport_speed_scale', 2, '--headless', '--diagnostic_events', '--diagnostic_interval', 25],
            'archived_checkpoint10_diagnostic')
        if code not in (0, 1):
            raise RuntimeError(f'Archived replay exited {code}')
        path = out/'evaluation.json'
        report = json.loads(path.read_text())
        close = json.loads((out/'environment_close.json').read_text())
        if (close.get('environment_closed') is not True or close.get('evaluation_complete') is not True
            or close.get('evaluation_sha256') != sha(path) or report.get('policy_sha256') != sha(policy)
            or report.get('seed') != 54321):
            raise ValueError('Archived replay hash/closure/seed mismatch')
        old = json.loads(reference.read_text())
        for a,b in zip(old['cases'], report['cases']):
            for key in ('source', 'target', 'jitter_mm', 'yaw_deg'):
                if a['reset'][key] != b['reset'][key]:
                    raise ValueError('Archived replay sampled different reset conditions')
        if len(old['cases']) != len(report['cases']):
            raise ValueError('Missing allocated episodes')
        audit = args.output/'receiver_close_audit.json'
        if run('autonomy_handover_audit.py', ['--run', out, '--policy', policy, '--output', audit], 'neural_contact_audit'):
            raise RuntimeError('Contact audit failed')
        changed = [i for i,(a,b) in enumerate(zip(old['cases'], report['cases'])) if a['success'] != b['success']]
        state.update(complete=True, stage='baseline_audited', archived_success=report['success'], episodes=16,
            original_success=old['success'], v51_success=v51['success'],
            physical_threshold_passed=distribution_gate(report,.5,2.),
            same_archived_implementation_changed_results=bool(changed), changed_episode_ids=changed,
            failed_episode_ids=[i for i,c in enumerate(report['cases']) if not c['success']],
            report=str(path), report_sha256=sha(path), contact_audit=str(audit),
            note='All attempts retained; this audit does not erase V51 failures or promote wider domain ranges.')
        save()
        return 0
    except BaseException as error:
        state.update(complete=False, stage='interrupted' if isinstance(error, KeyboardInterrupt) else 'workflow_error',
                     error=f'{type(error).__name__}: {error}')
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
