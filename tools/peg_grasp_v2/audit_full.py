"""Check completed full-transfer reports against their measured control traces."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from full_task import STAGES, secure_contact


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(report_path):
    report = json.loads(report_path.read_text())
    if not report['complete'] or report['error'] or report['interrupted']:
        raise ValueError('Report is incomplete or contains a runtime error')
    if report['full_success'] != report['requested_episodes']:
        raise ValueError('Only fully successful collections can pass the full-task audit')
    snapshot = Path(report['source_snapshot'])
    for name, digest in report['source_hashes'].items():
        if sha256(snapshot / name) != digest:
            raise ValueError(f'Changed experiment source: {name}')
    if report['donor_policy'] and sha256(Path(report['donor_policy'])) != report['donor_policy_sha256']:
        raise ValueError('Donor checkpoint changed')
    learned=report.get('full_task_learned_policy',False)
    if learned and sha256(Path(report['full_policy']))!=report['full_policy_sha256']:
        raise ValueError('Full neural checkpoint changed')
    measurements = {}
    carried = report['parameters'].get('peg_support_window', 'carried') == 'carried'
    assisted = report['assisted_grasp']
    with Path(report['trace']).open() as trace:
        for line in trace:
            row = json.loads(line)
            episode, stage = row['episode'], row['stage']
            case = measurements.setdefault(episode, dict(stages=[], samples=0,
                final_free_samples=0, maximum_finger_force_n=0., pressure_override_samples=0,
                receiver_preload_verified=False, receiver_preload_seconds=0., donor_pickup_unassisted=True,
                maximum_support_activation_displacement_mm=0.))
            if not case['stages'] or case['stages'][-1] != stage:
                case['stages'].append(stage)
            action = np.asarray(row.get('action', []))
            if action.shape != (16,) or not np.isfinite(action).all():
                raise ValueError('Missing or invalid pre-action command; cannot use this trace for learning')
            if len(row.get('pre_action_state', [])) != 2:
                raise ValueError('Missing measured pre-action state')
            if learned:
                obs=np.asarray(row.get('full_neural_observation',[]))
                motor=np.asarray(row.get('full_neural_action',[]))
                if obs.shape!=(76,) or motor.shape!=(5,) or not np.isfinite(obs).all() or not np.isfinite(motor).all():
                    raise ValueError('Missing finite neural inputs/actions in the full sequence')
                if not np.array_equal(obs[54:74],np.eye(len(STAGES))[STAGES.index(stage)]):
                    raise ValueError('Neural stage input differs from measured sequence')
            support = row.get('peg_support') or {}
            holder = support.get('holder', 0)
            if support.get('object_pose_writes') or support.get('object_velocity_writes'):
                raise ValueError('Support directly modifies object state')
            if support.get('break_events'):
                raise ValueError('Support joint broke during the experiment')
            if stage in STAGES[:7] and holder:
                case['donor_pickup_unassisted'] = False
                raise ValueError('The validated donor pickup must stay unassisted')
            if assisted:
                first = STAGES.index('donor_transfer' if carried else 'receiver_close')
                last = STAGES.index('receiver_release' if carried else 'receiver_hold')
                expected = (1 if STAGES.index(stage) < STAGES.index('donor_release') else 2)
                expected = expected if first <= STAGES.index(stage) < last else 0
                if holder != expected:
                    raise ValueError(f'Unexpected support holder in {stage}: {holder}, expected {expected}')
                if report['parameters']['peg_support_type'] == 'fixed' and holder and not support.get('joint_authored_enabled'):
                    raise ValueError('Fixed support holder has no enabled joint')
            elif holder:
                raise ValueError('An unassisted report contains an active support')
            if stage == 'receiver_close':
                case['receiver_preload_seconds'] = row['grip_control'][1]['preload_stable_s']
            if stage == 'donor_release' and row['step'] == 0:
                case['receiver_preload_verified'] = secure_contact(
                    row['pre_action_state'][1], report['parameters']['grip_force'])
                if not case['receiver_preload_verified'] or case['receiver_preload_seconds'] + 1e-9 < report['parameters']['preload_seconds']:
                    raise ValueError('Donor released without two measured opposing receiver contacts')
            if stage == 'donor_transfer' and row['step'] == 0 and holder:
                if not secure_contact(row['pre_action_state'][0], report['parameters']['grip_force']):
                    raise ValueError('Donor support acquired before measured wall contacts')
            if row['step'] == 0 and stage in ('donor_transfer', 'receiver_close', 'donor_release') and holder:
                before = np.asarray(row['pre_action_state'][0]['peg_root_w_m'])
                after = np.asarray(row['psm1']['peg_root_w_m'])
                delta = float(np.linalg.norm(after - before) * 1000)
                case['maximum_support_activation_displacement_mm'] = max(
                    case['maximum_support_activation_displacement_mm'], delta)
                if delta > 1.5:
                    raise ValueError('Support activation produced an excessive measured object displacement')
            forces = [f for arm in ('psm1', 'psm2') for f in row[arm]['finger_normal_force_n']]
            case['maximum_finger_force_n'] = max(case['maximum_finger_force_n'], max(forces))
            if max(forces) > report['parameters']['max_force']:
                raise ValueError('A successful trace exceeded the finger force limit')
            case['pressure_override_samples'] += bool(row.get('assisted_pressure_overrides'))
            if stage == 'placement_settle':
                if holder or not row['placement']['placed'] or max(forces) >= .01:
                    raise ValueError('Final placement is not free and physically settled')
                case['final_free_samples'] += 1
            case['samples'] += 1
    if len(measurements) != report['requested_episodes']:
        raise ValueError('Report and trace episode counts disagree')
    for case in report['cases']:
        measured = measurements[case['episode']]
        if measured['stages'] != list(STAGES) or not all(s['achieved'] for s in case['stages']):
            raise ValueError('Some full-task stages were skipped')
        if measured['final_free_samples'] < 100 or not measured['receiver_preload_verified']:
            raise ValueError('Missing physical handover or final free-placement verification')
        measured['placement'] = case['placement']
        measured['simulated_duration_s'] = measured['samples'] * .01
        measured['final_free_seconds'] = measured['final_free_samples'] * .01
    return dict(report=str(report_path), report_sha256=sha256(report_path),
        trace_sha256=sha256(Path(report['trace'])), passed=True,
        full_success=report['full_success'], assisted_grasp=assisted,
        full_task_learned_policy=report['full_task_learned_policy'],
        unassisted_full_success=report['unassisted_full_success'],
        source='L5', target='R2', device=report['device'],
        measurements=list(measurements.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.report)
    data = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as file:
            file.write(data)
    print(data)


if __name__ == '__main__':
    main()
