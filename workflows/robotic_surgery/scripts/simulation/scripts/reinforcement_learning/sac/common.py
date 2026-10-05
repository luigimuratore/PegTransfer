"""CLI, checkpoint provenance and construction shared by SAC entry points."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

TASK = 'Isaac-Peg-Transfer-Dual-PSM-SAC-v0'
SCHEMA = 'peg-sac-1'
SOURCE_LABELS = ('L5', 'L6', *(f'P{i}' for i in range(1, 10)))
REPO = Path(__file__).resolve().parents[7]
TASK_DIR = REPO / 'workflows/robotic_surgery/scripts/simulation/exts/robotic.surgery.tasks/robotic/surgery/tasks/surgical/peg_transfer'


def versions():
    return {p: importlib.metadata.version(p) for p in
            ('isaacsim', 'isaaclab', 'isaaclab_rl', 'torch', 'stable-baselines3', 'gymnasium')}


def fingerprint():
    return hashlib.sha256(b''.join(str(p.relative_to(REPO)).encode() + p.read_bytes() for p in source_files())).hexdigest()


def source_files():
    files = sorted(TASK_DIR.glob('sac_*.py')) + [TASK_DIR / 'agents/sac_cfg.py', TASK_DIR / 'env_cfg.py', TASK_DIR / 'scene_cfg.py']
    files += sorted(Path(__file__).parent.glob('*.py'))
    files += [REPO / 'workflows/robotic_surgery/scripts/simulation/utils/assets.py',
              REPO / 'workflows/robotic_surgery/scripts/simulation/exts/robotic.surgery.assets/robotic/surgery/assets/psm.py']
    return files


def asset_hashes():
    root = REPO / 'data/Isaac/Healthcare/0.5.0/132c82d/Props'
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / 'PegBlock/block.usd', root / 'Table/table.usd') if p.is_file()}


def add_launcher_options(p):
    # Importing AppLauncher itself bootstraps Kit in this installation. Keep --help
    # and checkpoint validation independent of all simulator imports.
    p.add_argument('--headless', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--enable_cameras', action='store_true')


def parser(mode):
    p = argparse.ArgumentParser(description=f'{mode} native SB3 SAC for two dVRK PSMs')
    p.add_argument('--num_envs', type=int, default=16 if mode != 'play' else 1)
    p.add_argument('--seed', type=int, default=42)
    # Keep the initial skill curriculum on one reproducible source. Expand the
    # source distribution only after the complete L5-to-R2 policy is validated.
    p.add_argument('--source', choices=('random', 'near', *SOURCE_LABELS), default='L5')
    if mode == 'train':
        p.add_argument('--phase', choices=('lift', 'handover', 'full'), default='lift')
        p.add_argument('--steps', type=int, default=1_000_000, help='Additional aggregate transitions')
        p.add_argument('--run_name', default=None)
        group = p.add_mutually_exclusive_group()
        group.add_argument('--resume', type=Path, help='Same phase: model + replay, same num_envs')
        group.add_argument('--initialize_from', type=Path, help='New phase/source curriculum: SAC policy/critics, fresh replay')
        p.add_argument('--save_every', type=int, default=100_000)
        p.add_argument('--buffer_size', type=int, default=500_000)
        p.add_argument('--learning_starts', type=int, default=10_000)
        p.add_argument('--gradient_steps', type=int, default=4)
        p.add_argument('--demonstrations', type=Path, help='Verified lift .npz: actor imitation and permanent demo replay')
        p.add_argument('--bc_steps', type=int, default=3000, help='Supervised actor updates before online SAC')
        p.add_argument('--bc_weight', type=float, default=1.0, help='Auxiliary actor imitation weight during SAC')
        p.add_argument('--demo_fraction', type=float, default=0.25, help='Demo share of each SAC replay minibatch')
        p.add_argument('--imitation_only', action='store_true', help='Save the initialized actor without online simulation steps')
    else:
        p.add_argument('--checkpoint', type=Path, required=True)
        p.add_argument('--video', action='store_true')
        p.add_argument('--video_length', type=int, default=1000)
        if mode == 'evaluate':
            p.add_argument('--episodes', type=int, default=100)
            p.add_argument('--output', type=Path, required=True)
            p.add_argument('--phase', choices=('lift', 'handover', 'full'), default='full',
                           help='Default full, even when checkpoint was trained only for lift')
        else:
            p.add_argument('--max_steps', type=int, default=0, help='0 = until window closes')
    add_launcher_options(p)
    args = p.parse_args()
    for name in ('num_envs', 'steps', 'episodes', 'save_every', 'buffer_size', 'gradient_steps', 'video_length'):
        if hasattr(args, name) and getattr(args, name) <= 0:
            p.error(f'--{name} must be positive')
    if getattr(args, 'learning_starts', 0) < 0 or getattr(args, 'max_steps', 0) < 0:
        p.error('learning_starts/max_steps must be nonnegative')
    if mode == 'train':
        if args.bc_steps < 0 or not 0 < args.bc_weight < float('inf') or not 0 < args.demo_fraction < 1:
            p.error('bc_steps must be nonnegative, bc_weight finite positive, and demo_fraction between 0 and 1')
        if args.demonstrations and (args.phase != 'lift' or args.resume):
            p.error('Demonstrations support lift training; resume restores its existing demo configuration')
        if args.imitation_only and not args.demonstrations:
            p.error('--imitation_only requires --demonstrations on a fresh lift run')
    if getattr(args, 'video', False):
        args.enable_cameras = True
    return args


def read_checkpoint(path):
    path = path.resolve()
    if path.suffix != '.zip' or not path.is_file():
        raise ValueError('Provide an existing SAC .zip checkpoint; PPO .pt cannot be loaded as SAC')
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta.get('algorithm') != 'SAC' or meta.get('schema') != SCHEMA or meta.get('task') != TASK:
        raise ValueError('Checkpoint algorithm/task/schema mismatch')
    if meta['fingerprint'] != fingerprint():
        raise ValueError('Task or pipeline code changed since training; checkpoint requires a matching checkout')
    if meta.get('assets', asset_hashes()) != asset_hashes():
        raise ValueError('Local USD assets changed since training')
    return meta


def save_checkpoint(model, path, manifest, replay=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    if replay:
        model.save_replay_buffer(path.with_name(path.stem + '_replay.pkl'))
    meta = dict(manifest, timesteps=model.num_timesteps, replay_saved=replay)
    path.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')


def make_env(args, phase):
    import gymnasium as gym
    import robotic.surgery.tasks  # noqa: F401: task registration after AppLauncher
    from isaaclab_tasks.utils import parse_env_cfg
    from robotic.surgery.tasks.surgical.peg_transfer.sac_env_cfg import configure_phase
    from vec_env import PegSACVecEnv
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    configure_phase(cfg, phase)
    cfg.seed = args.seed
    cfg.source_post = args.source
    raw = gym.make(TASK, cfg=cfg, render_mode='rgb_array' if getattr(args, 'video', False) else None)
    if getattr(args, 'video', False):
        video_dir = args.checkpoint.resolve().parent / 'videos' / 'sac'
        raw = gym.wrappers.RecordVideo(raw, video_folder=str(video_dir), step_trigger=lambda s: s == 0,
                                       video_length=args.video_length, disable_logger=True)
    return PegSACVecEnv(raw), cfg
