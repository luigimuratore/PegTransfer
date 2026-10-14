"""Physically screen a wider reset curriculum with frozen checkpoint10 weights.

No training or promotion to a new permanent checkpoint happens here. Stop at
the first failed gate and retain failed cases for targeted BC/DAgger recovery.
Each post pair is tested explicitly; a global rate cannot hide an unseen pair.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from autonomy_constraints_gate import assisted_gate
from autonomy_motion_curriculum import sha, speed_comparison
from autonomy_env import SOURCES, TARGETS
from geometry import load_profile, validate_reset_range
from preserve_autonomy import verify_archive

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SEEDS = (2468, 54321)


def curriculum_levels():
    levels = [dict(name='baseline', sources=['L5'], targets=['R2'], jitter_mm=.5, yaw_deg=2.)]
    for yaw in (5., 10., 20., 30.):
        levels.append(dict(name=f'yaw{yaw:g}', sources=['L5'], targets=['R2'], jitter_mm=.5, yaw_deg=yaw))
    for jitter in (.75, 1., 1.1):
        levels.append(dict(name=f'offset{jitter:g}', sources=['L5'], targets=['R2'], jitter_mm=jitter, yaw_deg=30.))
    # Reduce local variation when introducing a new discrete post. These are
    # planned, not certified; the physical gate may stop well before all posts.
    for name, sources, targets in (
        ('two_targets', ['L5'], ['R2', 'R1']),
        ('two_sources', ['L5', 'L6'], ['R2', 'R1']),
        ('inner_posts', ['L5', 'L6', 'L4'], ['R2', 'R1', 'R3', 'R4']),
        ('all_posts', ['L5', 'L6', 'L4', 'L2', 'L1', 'L3'], ['R2', 'R1', 'R3', 'R4', 'R5', 'R6']),
        ('all_posts_local', ['L5', 'L6', 'L4', 'L2', 'L1', 'L3'], ['R2', 'R1', 'R3', 'R4', 'R5', 'R6']),
    ):
        levels.append(dict(name=name, sources=sources, targets=targets,
                           jitter_mm=.5 if name == 'all_posts_local' else .25,
                           yaw_deg=5. if name == 'all_posts_local' else 2.))
    return levels


def pair_gate(report, source, target, jitter, yaw, count=16):
    if source not in SOURCES or target not in TARGETS:
        return False
    if not assisted_gate(report, 1., count, 1. if count == 3 else .9, 'firm', True):
        return False
    config = report.get('reset_distribution', {})
    if (config.get('sources') != [source] or config.get('targets') != [target]
        or config.get('jitter_mm') != jitter or config.get('yaw_deg') != yaw
        or config.get('reset_only') is not True):
        return False
    speed = report.get('motion_reference', {})
    if (speed.get('profile') != 'travel' or speed.get('free_scale') != 2.
        or speed.get('transport_scale') != 2. or speed.get('contact_scale') != 1.):
        return False
    for row in report['cases']:
        reset = row.get('reset', {})
        offset, post = reset.get('jitter_mm', []), reset.get('source_post_center_m', [])
        expected = [*SOURCES[source], .01]
        destination = reset.get('target_post_center_m', [])
        expected_target = [*TARGETS[target], .01]
        if (reset.get('source') != source or reset.get('target') != target
            or len(offset) != 2 or any(not math.isfinite(v) or abs(v) > jitter+1e-5 for v in offset)
            or not math.isfinite(reset.get('yaw_deg', float('nan'))) or abs(reset['yaw_deg']) > yaw+1e-5
            or len(post) != 3 or any(not math.isfinite(a) or abs(a-b) > 1e-7 for a,b in zip(post, expected))
            or len(destination) != 3 or any(not math.isfinite(a) or abs(a-b) > 1e-7 for a,b in zip(destination, expected_target))
            or reset.get('reset_only_randomization') is not True
            or reset.get('board_and_posts_unchanged') is not True):
            return False
    return True


def level_gate(rows, level):
    expected = {(source, target, seed) for source in level['sources']
                for target in level['targets'] for seed in SEEDS}
    seen = {(r['source'], r['target'], r['seed']) for r in rows}
    return (seen == expected and len(rows) == len(expected)
            and all(r['passed'] and r['jitter_mm'] == level['jitter_mm']
                    and r['yaw_deg'] == level['yaw_deg'] for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--through', choices=[x['name'] for x in curriculum_levels()], default='all_posts_local')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if not args.device.startswith('cuda'):
        p.error('CUDA required')
    milestone = verify_archive(ROOT/'milestones/checkpoint10-assisted-full-bc-transit.json')
    parent = ROOT/milestone['preserved_policy']
    archive = ROOT/milestone['preserved_bundle']
    levels = curriculum_levels()
    levels = levels[:next(i for i,x in enumerate(levels) if x['name'] == args.through)+1]
    for level in levels:
        validate_reset_range(load_profile(), level['jitter_mm'], level['yaw_deg'])
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output/'workflow_source.py')
    state = dict(complete=False, stage='freeze_runtime', algorithm='Frozen BC generalization screening; no training',
                 parent_checkpoint='checkpoint10', policy_sha256=sha(parent), levels=levels,
                 results=[], verified_levels=[], assisted_grasp=True, native_full_policy_verified=False,
                 gate='At least 15/16 on each seed and each explicit post pair; baseline retains every checkpoint10 success',
                 view='Front camera: L1..L6 on image right for picking; R1..R6 on image left for placement')
    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        tmp = args.output/'status.json.tmp'
        tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(args.output/'status.json')
    def run(module, argv, label):
        state['stage'] = label
        save()
        print(f'[DOMAIN] {label}', flush=True)
        with (args.output/f'{label}.log').open('w') as stream:
            child = subprocess.Popen([sys.executable, '-u', str(HERE/module), *map(str, argv)],
                                     cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
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
        if code not in (0, 1):
            raise RuntimeError(f'{label} exited {code}; see its log')
        return code
    try:
        policy = args.output/'runtime/policy.pt'
        code = run('autonomy_main.py', ['repackage', '--policy', parent, '--output', policy.parent,
            '--skill', 'full', '--device', args.device, '--implementation_change',
            'Expanded actual post names and mesh-screened reset range; exact checkpoint10 weights and motor speeds'], 'freeze_runtime')
        if code or sha(policy) != sha(parent):
            raise ValueError('Repackaging changed the frozen weights or failed')
        state['runtime_policy'] = str(policy)
        save()
        for level in levels:
            rows = []
            for source in level['sources']:
                for target in level['targets']:
                    for seed in SEEDS:
                        label = f"{level['name']}_{source}_{target}_seed{seed}"
                        out = args.output/label
                        code = run('autonomy_replay.py', ['evaluate', '--policy', policy, '--output', out,
                            '--skill', 'full', '--num_envs', 16, '--episodes', 16, '--episode_steps', 10000,
                            '--device', args.device, '--seed', seed, '--sources', source, '--targets', target,
                            '--jitter_mm', level['jitter_mm'], '--yaw_deg', level['yaw_deg'],
                            '--grasp_support', 1, '--support_profile', 'firm', '--placement_support',
                            '--motion_profile', 'travel', '--free_speed_scale', 2, '--transport_speed_scale', 2,
                            '--headless'], label)
                        path = out/'evaluation.json'
                        report = json.loads(path.read_text())
                        close = json.loads((out/'environment_close.json').read_text())
                        if (close.get('environment_closed') is not True or close.get('evaluation_complete') is not True
                            or close.get('evaluation_sha256') != sha(path) or report.get('policy_sha256') != sha(policy)
                            or report.get('seed') != seed):
                            raise ValueError('Unverified weights, report, seed or environment cleanup')
                        passed = pair_gate(report, source, target, level['jitter_mm'], level['yaw_deg'])
                        if level['name'] == 'baseline':
                            reference = json.loads((archive/f'reports/transit0_seed{seed}.json').read_text())
                            passed = passed and speed_comparison(reference, report, min_gain=-.001)
                        row = dict(label=label, source=source, target=target, seed=seed, passed=passed,
                                   success=report['success'], episodes=16, jitter_mm=level['jitter_mm'], yaw_deg=level['yaw_deg'],
                                   report=str(path), report_sha256=sha(path))
                        state['results'].append(row)
                        rows.append(row)
                        save()
                        print(f"[DOMAIN] {label}: {report['success']}/16 passed={passed}", flush=True)
                        if passed and code:
                            raise ValueError('Passed physical gate but process exited unsuccessfully')
                        # Test both seeds of the failed pair to retain useful recovery cases.
                    if not all(r['passed'] for r in rows):
                        failed = [r for r in rows if not r['passed']]
                        state.update(complete=True, stage='domain_gate_failed', failed_level=level,
                                     failed_reports=failed, training_required=True,
                                     reason='Range not promoted. Collect targeted BC/DAgger recovery at this level and recheck prior conditions; checkpoint10 retained.')
                        save()
                        return 1
            if not level_gate(rows, level):
                raise ValueError('Missing explicit pair/seed coverage')
            state['verified_levels'].append(level)
            save()
        state.update(complete=True, stage='requested_domain_verified', training_required=False)
        save()
        return 0
    except BaseException as error:
        state.update(complete=False, stage='interrupted' if isinstance(error, KeyboardInterrupt) else 'workflow_error',
                     error=f'{type(error).__name__}: {error}')
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
