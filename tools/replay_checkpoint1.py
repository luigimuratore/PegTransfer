"""Preserve/replay checkpoint1; prepare and check never launch Isaac Sim."""
import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'milestones/checkpoint1.json'
CACHE = ROOT / '.checkpoint_runs/checkpoint1'
RL_REL = Path('workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/sac_sequence')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(record):
    if CACHE.exists():
        return CACHE
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='checkpoint1-', dir=CACHE.parent))
    try:
        paths = ['workflows', 'tools', 'peg_sequence.sh', str(Path(record['checkpoint']).parent)]
        archive = subprocess.check_output(
            ['git', 'archive', '--format=tar', record['commit'], *paths], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            for member in bundle.getmembers():
                target = (temp / member.name).resolve()
                if not target.is_relative_to(temp.resolve()) or not (member.isfile() or member.isdir()):
                    raise ValueError('Unexpected archive entry; snapshot not extracted')
            bundle.extractall(temp, filter='data')
        # Freeze the locally downloaded assets, including textures/references.
        # Later edits to ROOT/data must not change this milestone.
        if not (ROOT / 'data').is_dir():
            raise ValueError('Local data/ assets are missing. Restore the checkpoint1 assets before preparing it.')
        shutil.copytree(ROOT / 'data', temp / 'data')
        (temp / 'snapshot.json').write_text(json.dumps(record, indent=2) + '\n')
        temp.rename(CACHE)
        return CACHE
    finally:
        if temp.exists():
            shutil.rmtree(temp)  # Only this function's incomplete temporary directory.


def frozen_environment(snapshot):
    scripts = snapshot / 'workflows/robotic_surgery/scripts'
    exts = scripts / 'simulation/exts'
    env = os.environ.copy()
    # Namespace packages must resolve to the frozen task/assets, ahead of the
    # editable packages installed from the working project.
    paths = [str(exts / 'robotic.surgery.tasks'), str(exts / 'robotic.surgery.assets'), str(scripts)]
    paths += [p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p]
    env['PYTHONPATH'] = os.pathsep.join(paths)
    return env


def verify(snapshot, record):
    if json.loads((snapshot / 'snapshot.json').read_text()) != record:
        raise ValueError('Snapshot metadata differs; keep the original checkpoint1 record and snapshot.')
    checkpoint = snapshot / record['checkpoint']
    if sha256(checkpoint) != record['checkpoint_sha256']:
        raise ValueError('The preserved model has changed. Restore it from the pinned Git commit.')
    meta = json.loads(checkpoint.with_suffix('.json').read_text())
    if meta['fingerprint'] != record['fingerprint']:
        raise ValueError('Checkpoint metadata does not match checkpoint1.')
    for name, expected in record['versions'].items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f'{name}: installed {actual}, checkpoint1 requires {expected}. Use the original PegTransfer environment.')
    # Execute only the original preflight checks: no AppLauncher, no simulation.
    code = '''
import importlib.util, json, sys
from pathlib import Path
root = Path(sys.argv[1])
rl = root / sys.argv[2]
sys.path.insert(0, str(rl))
from common import checkpoint
meta = checkpoint(root / sys.argv[3])
for package in ('robotic.surgery.tasks', 'robotic.surgery.assets', 'simulation'):
    spec = importlib.util.find_spec(package)
    locations = [spec.origin] if spec.origin else list(spec.submodule_search_locations or [])
    if not locations or not all(Path(p).resolve().is_relative_to(root) for p in locations):
        raise ValueError('Package resolves outside checkpoint1: ' + package)
print(json.dumps({'fingerprint': meta['fingerprint'], 'assets': meta['assets']}))
'''
    subprocess.run([sys.executable, '-c', code, str(snapshot), str(RL_REL), record['checkpoint']],
                   cwd=snapshot, env=frozen_environment(snapshot), check=True)
    print('[CHECKPOINT1] Original code, model, local assets and library versions verified.', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Replay checkpoint1 using its original task, model and local assets.')
    parser.add_argument('mode', choices=('prepare', 'check', 'play', 'video', 'eval'), nargs='?', default='check')
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.episodes is not None and args.episodes < 1:
        parser.error('--episodes must be positive')
    record = json.loads(RECORD.read_text())
    snapshot = prepare(record)
    verify(snapshot, record)
    print(f'[CHECKPOINT1] Frozen task: {snapshot}', flush=True)
    if args.mode in ('prepare', 'check'):
        return
    output_dir = ROOT / 'logs/checkpoint1'
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f'{args.mode}_{time.strftime("%Y-%m-%d_%H-%M-%S")}_{time.time_ns()}.json'
    evaluation = args.mode == 'eval'
    entry = snapshot / RL_REL / ('evaluate.py' if evaluation else 'play.py')
    episodes = args.episodes or (100 if evaluation else 1 if args.mode == 'video' else 3)
    command = [sys.executable, str(entry), '--checkpoint', str(snapshot / record['checkpoint']),
               '--source', 'L5', '--skill', 'full', '--guidance', '0',
               '--num_envs', '16' if evaluation else '1', '--episodes', str(episodes),
               '--seed', str(args.seed), '--device', args.device, '--output', str(output)]
    if evaluation or args.mode == 'video':
        command += ['--headless']
    if args.mode == 'video':
        command += ['--video', '--trace']
    os.chdir(snapshot)
    os.execve(sys.executable, command, frozen_environment(snapshot))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError, importlib.metadata.PackageNotFoundError) as error:
        print(f'[CHECKPOINT1] {error}', file=sys.stderr)
        sys.exit(1)
