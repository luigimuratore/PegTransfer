"""Sequence-v1 provenance and CLI. No Isaac imports until make_env()."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[6]
SCRIPTS = REPO/'workflows/robotic_surgery/scripts'
sys.path.insert(0, str(SCRIPTS))
SURGICAL = REPO/'workflows/robotic_surgery/scripts/simulation/exts/robotic.surgery.tasks/robotic/surgery/tasks/surgical'
TASK = 'Isaac-Peg-Transfer-Dual-PSM-Sequence-v1'
SCHEMA = 'peg-sequence-sac-1'
SKILLS = ('approach', 'grasp', 'lift', 'transfer', 'receive', 'handover', 'full')

def sources():
    return sorted(HERE.glob('*.py')) + sorted((SURGICAL/'peg_transfer_sequence').glob('*.py')) + [
        SURGICAL/'peg_transfer'/name for name in ('sac_mdp.py', 'sac_env.py', 'sac_env_cfg.py',
            'sac_actions.py', 'sac_assets.py', 'sac_geometry.py', 'env_cfg.py', 'scene_cfg.py')] + [
        SCRIPTS/'simulation/utils/assets.py',
        SCRIPTS/'simulation/exts/robotic.surgery.assets/robotic/surgery/assets/psm.py']

def fingerprint():
    return hashlib.sha256(b''.join(str(p.relative_to(REPO)).encode()+p.read_bytes() for p in sources())).hexdigest()

def assets():
    paths = [REPO/'data/Isaac/Healthcare/0.5.0/132c82d/Props'/name for name in ('PegBlock/block.usd', 'Table/table.usd')]
    paths += [REPO/'data/Isaac/Healthcare/0.5.0/132c82d/Robots/dVRK/PSM/psm.usd']
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}

def versions():
    return {name: importlib.metadata.version(name) for name in
            ('isaacsim', 'isaaclab', 'torch', 'stable-baselines3', 'gymnasium')}

def checkpoint(path):
    path = Path(path).resolve()
    if not path.is_file() or path.suffix != '.zip':
        raise ValueError('Provide an existing sequence-v1 SAC .zip checkpoint')
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta.get('schema') != SCHEMA or meta.get('task') != TASK:
        raise ValueError('Legacy PPO/SAC checkpoints have a different policy/action/observation schema; start a new run')
    if meta['fingerprint'] != fingerprint() or meta['assets'] != assets():
        raise ValueError('Code/assets changed since this checkpoint; use the matching source snapshot')
    return meta

def save(model, path, manifest, replay=False):
    path = Path(path)
    model.save(path)
    if replay:
        model.save_replay_buffer(path.with_name(path.stem+'_replay.pkl'))
    path.with_suffix('.json').write_text(json.dumps(dict(manifest, timesteps=model.num_timesteps), indent=2)+'\n')

def cli(mode):
    p = argparse.ArgumentParser(description=f'{mode}: measured dual-PSM sequence SAC v1')
    p.add_argument('--source', choices=('L5', 'L6'), default='L5')
    p.add_argument('--skill', choices=SKILLS, default='full')
    p.add_argument('--guidance', type=float, default=None, help='1=analytic baseline+small residual; 0=learned active-arm motor commands')
    p.add_argument('--num_envs', type=int, default=16 if mode in ('train', 'evaluate') else 1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--enable_cameras', action='store_true')
    p.add_argument('--episode_seconds', type=float, default=60.)
    p.add_argument('--stage_budget', type=int, default=600)
    p.add_argument('--video', action='store_true')
    p.add_argument('--video_length', type=int, default=3000)
    if mode == 'train':
        p.add_argument('--initialize_from', type=Path)
        p.add_argument('--demonstrations', type=Path, required=True)
        p.add_argument('--run_name', default='sequence')
        p.add_argument('--steps', type=int, default=100000)
        p.add_argument('--bc_steps', type=int, default=5000)
        p.add_argument('--bc_weight', type=float, default=5.)
        p.add_argument('--learning_rate', type=float, default=1e-4)
        p.add_argument('--batch_size', type=int, default=256)
        p.add_argument('--buffer_size', type=int, default=300000)
        p.add_argument('--gradient_steps', type=int, default=1)
        p.add_argument('--demo_fraction', type=float, default=.5)
        p.add_argument('--ent_coef', type=float, default=.001, help='Fixed low exploration temperature; tune only with gated evaluations')
        p.add_argument('--imitation_only', action='store_true')
        p.add_argument('--save_every', type=int, default=25000)
    else:
        p.add_argument('--checkpoint', type=Path)
        p.add_argument('--baseline', action='store_true', help='Run analytic controller; never label this a learned policy')
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--episodes', type=int, default=100 if mode == 'evaluate' else 3)
        p.add_argument('--trace', action='store_true')
        if mode == 'collect':
            p.add_argument('--demonstrations', type=Path, required=True)
            p.add_argument('--guidance_levels', type=float, nargs='+', default=[1., .5, 0.])
            p.add_argument('--action_noise', type=float, default=.015)
    args = p.parse_args()
    if args.guidance is not None and not 0 <= args.guidance <= 1:
        p.error('--guidance must be in [0,1]')
    for key in ('num_envs', 'stage_budget', 'episode_seconds', 'episodes', 'steps', 'batch_size', 'buffer_size',
                'gradient_steps', 'save_every', 'video_length', 'learning_rate'):
        if hasattr(args, key) and not getattr(args, key) > 0:
            p.error(f'--{key} must be positive')
    if mode == 'train' and (args.bc_steps < 0 or not 0 < args.demo_fraction < 1 or args.bc_weight < 0 or args.ent_coef < 0):
        p.error('Invalid imitation/replay/entropy setting')
    if mode == 'collect':
        if args.num_envs != 1 or args.skill != 'full':
            p.error('Collection requires --num_envs 1 --skill full')
        if args.action_noise < 0 or any(not 0 <= level <= 1 for level in args.guidance_levels):
            p.error('Invalid guidance levels/noise')
        if args.demonstrations.suffix != '.npz' or args.demonstrations.exists() or args.demonstrations.with_suffix('.json').exists():
            p.error('Choose a new .npz demonstration archive')
    if mode in ('evaluate', 'collect', 'probe', 'play'):
        if args.output.exists() or args.output.with_suffix('.trace.jsonl').exists() or args.output.with_suffix('.episodes.jsonl').exists():
            p.error('Choose a new output filename to preserve earlier results')
        if mode != 'collect' and bool(args.checkpoint) == bool(args.baseline):
            p.error('Choose exactly one of --checkpoint and --baseline')
    if args.video:
        args.enable_cameras = True
    return args

def make_env(args):
    import gymnasium as gym
    import robotic.surgery.tasks  # Registration after AppLauncher only.
    from isaaclab_tasks.utils import parse_env_cfg
    from adapter import SequenceVecEnv
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.source_post, cfg.skill, cfg.guidance = args.source, args.skill, args.guidance
    cfg.seed, cfg.stage_budget, cfg.episode_length_s = args.seed, args.stage_budget, args.episode_seconds
    raw = gym.make(TASK, cfg=cfg, render_mode='rgb_array' if args.video else None)
    if args.video:
        directory = args.output.parent/'sequence_videos'/args.output.stem
        raw = gym.wrappers.RecordVideo(raw, video_folder=str(directory), step_trigger=lambda step: step == 0,
                                       video_length=args.video_length, disable_logger=True)
    return SequenceVecEnv(raw), cfg
