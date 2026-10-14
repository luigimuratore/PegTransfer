"""Replay the preserved assisted full-task teacher, without changing its physics."""
import argparse
from datetime import datetime
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]


def verify(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    bundle = REPO / manifest['bundle_root']
    for name, digest in manifest['files_sha256'].items():
        path = bundle / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Preserved file is missing or changed: {path}')
    for name, version in manifest['versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Original environment required: {name}=={version}')
    return manifest, bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=REPO / 'milestones/checkpoint3-full-assisted.json')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--realtime', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--video_path', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest, bundle = verify(args.manifest)
    if args.check:
        print(f"Preserved teacher verified: {manifest['full_success']}/{manifest['requested_episodes']}, assisted grasp")
        return 0
    if args.episodes < 1 or (args.video and args.episodes != 1):
        parser.error('Use positive episodes and one episode for video')
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    output = (args.output or REPO / f'logs/grasp_v2/full_assisted_replay_{stamp}.json').resolve()
    argv = ['--donor_policy', str(bundle / 'donor_checkpoint/policy.npz'),
        '--robot_usd', str(bundle / 'data/Isaac/Healthcare/0.5.0/132c82d/Robots/dVRK/PSM/psm.usd'),
        '--episodes', str(args.episodes), '--device', args.device, '--output', str(output)]
    for key, value in manifest['parameters'].items():
        if isinstance(value, bool):
            if value: argv.append('--' + key)
        else:
            argv += ['--' + key] + [str(v) for v in (value if isinstance(value, list) else [value])]
    if args.headless or args.video: argv.append('--headless')
    if args.realtime: argv.append('--realtime')
    video = (args.video_path or REPO / f'logs/grasp_v2/videos/full_assisted_front_{stamp}.mp4').resolve()
    if args.video: argv += ['--video_path', str(video)]
    scripts = bundle / 'workflows/robotic_surgery/scripts'
    extensions = scripts / 'simulation/exts'
    sys.path[:0] = [str(bundle / 'tools/peg_grasp_v2'), str(scripts),
        str(extensions / 'robotic.surgery.tasks'), str(extensions / 'robotic.surgery.assets')]
    full = importlib.import_module('full_task')
    full.REPO = bundle
    if args.video:
        replay = importlib.import_module('replay')
        original = replay.EpisodeVideo
        def closed(recorder):
            report = json.loads(output.read_text())
            metadata = dict(manifest=str(args.manifest), video=str(video), report=str(output),
                full_success=report['full_success'], assisted_grasp=report['assisted_grasp'],
                full_task_learned_policy=report['full_task_learned_policy'],
                encoder='h264_nvenc', frames=recorder.frames, fps=25,
                duration_s=recorder.frames / 25, resolution=[1280,720],
                camera=report['camera'], video_sha256=hashlib.sha256(video.read_bytes()).hexdigest())
            with video.with_suffix('.json').open('x') as file:
                json.dump(metadata, file, indent=2); file.write('\n')
            print(f'[FULL VIDEO] {video}; {recorder.frames} frames', flush=True)
        replay.EpisodeVideo = lambda env, path: original(env, path, on_closed=closed)
    sys.argv = [str(bundle / 'tools/peg_grasp_v2/full_task.py'), *argv]
    return full.main()


if __name__ == '__main__':
    raise SystemExit(main())
