"""Inspect recorded receiver closure without changing the actor or simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run, policy, device='cuda:0'):
    # Use the exact implementation which produced the physical observations.
    manifest = json.loads((policy.parent / 'manifest.json').read_text())
    if digest(policy) != manifest['policy_sha256']:
        raise ValueError('Frozen policy hash differs from its manifest')
    sources = policy.parent / 'sources'
    for name, expected in manifest['source_hashes'].items():
        if digest(sources / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    sys.path.insert(0, str(sources.resolve()))
    import torch
    from autonomy_learning import load_policy
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('CUDA required for neural diagnostics')
    torch.set_num_threads(8)
    net = load_policy(policy, device)
    if net.jaw_rate_head is None:
        raise ValueError('A learned jaw rate head is required')
    observations, actions, goals, episodes, steps = [], [], [], [], []
    trace = run / 'policy_events.jsonl'
    for line in trace.read_text().splitlines():
        event = json.loads(line)
        size = len(event['episode_ids'])
        for name in ('options', 'physical_observations', 'actions', 'learned_goals'):
            if len(event[name]) != size:
                raise ValueError('Mismatched diagnostic event arrays')
        for i, option in enumerate(event['options']):
            if option != 10:
                continue
            observations.append(event['physical_observations'][i])
            actions.append(event['actions'][i])
            goals.append(event['learned_goals'][i])
            episodes.append(event['episode_ids'][i])
            steps.append(event['step'])
    if not observations:
        raise ValueError('No recorded receiver-close observations yet')
    obs = torch.tensor(observations, device=device)
    action = torch.tensor(actions, device=device)
    goal = torch.tensor(goals, device=device).reshape(-1, 2, 8)
    episode = torch.tensor(episodes, device=device)
    if obs.shape[1] != 98 or action.shape[1] != 14:
        raise ValueError('Incompatible diagnostic dimensions')
    if not all(bool(torch.isfinite(x).all()) for x in (obs, action, goal)):
        raise ValueError('Nonfinite physical diagnostics')
    with torch.inference_mode():
        option = torch.full((len(obs),), 10, dtype=torch.long, device=device)
        head = getattr(net, 'receiver_jaw_rate_head', None)
        if head is None or not bool(net.receiver_jaw_options[10]):
            head = net.jaw_rate_head
        logits = head(obs, option, logits=True)[:, 1]
        probability = logits.softmax(-1)
        rate = head.rates[logits.argmax(-1)] * .15
        forces = obs[:, 56:62].reshape(-1, 2, 3)
        norms = forces.norm(dim=-1)
        cosine = (forces[:, 0] * forces[:, 1]).sum(-1) / norms.prod(-1).clamp_min(1e-8)
        goal_error = ((goal[:, 1, :3] - obs[:, 35:38]) * 100).norm(dim=-1)
        # Actual emitted absolute jaw target compared with the current reference.
        commanded_change = (action[:, 13] + 1) * .25 - obs[:, 69] * .5
        rows = []
        for ep in episode.unique().tolist():
            ids = (episode == ep).nonzero().flatten()
            count = len(ids)
            last = int(ids[-1])
            rows.append(dict(
                episode_id=ep, recorded_samples=count,
                first_step=steps[int(ids[0])], last_step=steps[last],
                receiver_force_min_n=float(norms[ids].amin()),
                last_receiver_forces_n=norms[last].tolist(),
                last_force_opposition_cosine=float(cosine[last]),
                last_receiver_jaws_rad=(obs[last, 54:56] * .5).tolist(),
                last_neural_jaw_rate_rad_s=float(rate[last]),
                last_jaw_class_probabilities=probability[last].tolist(),
                sampled_neural_close_fraction=float((rate[ids] < 0).float().mean()),
                sampled_neural_hold_fraction=float((rate[ids] == 0).float().mean()),
                sampled_emitted_close_fraction=float((commanded_change[ids] < -1e-6).float().mean()),
                last_receiver_neural_goal_error_mm=float(goal_error[last]),
                last_physical_progress=obs[last, 90:94].tolist()))
    report_path = run / 'evaluation.json'
    complete = report_path.exists() and json.loads(report_path.read_text()).get('complete') is True
    physical = json.loads(report_path.read_text()) if complete else {}
    if complete:
        close = json.loads((run / 'environment_close.json').read_text())
        if (physical.get('policy_sha256') != digest(policy)
                or close.get('environment_closed') is not True
                or close.get('evaluation_sha256') != digest(report_path)):
            raise ValueError('Physical report/policy or clean-closure receipt mismatch')
    cases = physical.get('cases', [])
    for row in rows:
        if row['episode_id'] < len(cases):
            case = cases[row['episode_id']]
            row['physical_result'] = {key: case.get(key) for key in
                ('success', 'timeout', 'failure', 'grasp_psm1', 'lift_stable', 'grasp_psm2', 'handover', 'terminal_learned_option')}
    return dict(diagnostic_only=True, complete_physical_evaluation=complete,
        policy_sha256=digest(policy), trace_sha256=digest(trace),
        source_hashes_verified=True, neural_computation_device=device,
        scope='Receiver close, learned option 10; logged samples, not all physics substeps',
        jaw_class_rates_rad_s=(head.rates * .15).tolist(),
        cases=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait_minutes', type=float, default=0,
                        help='Wait for physical evaluation and clean closure before auditing')
    args = parser.parse_args()
    if args.wait_minutes < 0:
        parser.error('Waiting limit cannot be negative')
    status_path = args.run / 'diagnostic_status.json'
    def status(**values):
        values['updated_utc'] = datetime.now(timezone.utc).isoformat()
        values['diagnostic_only'] = True
        values['policy'] = str(args.policy)
        values['run'] = str(args.run)
        temporary = status_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(values, indent=2) + '\n')
        temporary.replace(status_path)
    try:
        if args.wait_minutes:
            status(complete=False, stage='waiting_for_physical_evaluation')
            deadline = time.monotonic() + args.wait_minutes * 60
            while not (args.run / 'environment_close.json').exists():
                error = args.run / 'error.json'
                if error.exists():
                    raise RuntimeError(error.read_text())
                if time.monotonic() > deadline:
                    raise TimeoutError('Physical evaluation has not closed before the waiting limit')
                time.sleep(10)
        report = audit(args.run, args.policy)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        status(complete=report['complete_physical_evaluation'], stage='receiver_close_audit_saved',
               audit=str(args.output), native_full_policy_verified=False,
               next_step='Inspect receiver closure before supervised recovery training; keep V42 accepted speed/reset range')
    except BaseException as exc:
        status(complete=False, stage='diagnostic_error', error=f'{type(exc).__name__}: {exc}')
        raise
    for row in report['cases']:
        print(f"episode={row['episode_id']} last_step={row['last_step']} "
              f"forces={row['last_receiver_forces_n']} jaw_rate={row['last_neural_jaw_rate_rad_s']:.3f} "
              f"hold_samples={row['sampled_neural_hold_fraction']:.2f} "
              f"goal_error={row['last_receiver_neural_goal_error_mm']:.3f}mm")


if __name__ == '__main__':
    main()
