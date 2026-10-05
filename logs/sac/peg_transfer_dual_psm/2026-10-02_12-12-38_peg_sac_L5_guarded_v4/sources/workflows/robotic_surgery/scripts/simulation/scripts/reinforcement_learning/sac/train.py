"""Train native Stable-Baselines3 SAC; never load an RSL-RL/PPO checkpoint."""

from datetime import datetime
import json
import subprocess
import shutil
from pathlib import Path

from common import REPO, SCHEMA, TASK, asset_hashes, fingerprint, make_env, parser, read_checkpoint, save_checkpoint, source_files, versions


def main():
    args = parser('train')
    prior = args.resume or args.initialize_from
    meta = read_checkpoint(prior) if prior else None  # reject PPO before launching Sim
    if args.resume and (meta['phase'] != args.phase or meta['num_envs'] != args.num_envs
                        or meta.get('source', 'random') != args.source):
        raise ValueError('Resume requires same phase, source and num_envs; use initialize_from to change curriculum')
    replay_path = prior.with_name(prior.stem + '_replay.pkl') if args.resume else None
    if replay_path and not replay_path.is_file():
        raise FileNotFoundError(f'Resume requires replay buffer: {replay_path}; use a final checkpoint')
    demo_meta, demo_data = None, None
    demo_path = args.demonstrations
    if args.resume and meta.get('demonstrations'):
        demo_path = meta['demonstrations']['path']
    if demo_path:
        from demonstrations import load_demonstrations
        demo_data, demo_meta = load_demonstrations(demo_path, args.phase)
        if 'L5' not in set(demo_data['sources']):
            raise ValueError('The staged lift curriculum requires successful L5 demonstrations')
        if args.resume and demo_meta['sha256'] != meta['demonstrations']['sha256']:
            raise ValueError('Resume requires the same demonstration archive')
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    env = None
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback
        from isaaclab.utils.io import dump_yaml
        from robotic.surgery.tasks.surgical.peg_transfer.agents.sac_cfg import SAC_SETTINGS
        env, cfg = make_env(args, args.phase)
        run = args.run_name or f'peg_sac_{args.phase}'
        log = REPO / 'logs/sac/peg_transfer_dual_psm' / (datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + run)
        log.mkdir(parents=True, exist_ok=False)
        settings = dict(SAC_SETTINGS, buffer_size=args.buffer_size, learning_starts=args.learning_starts,
                        gradient_steps=args.gradient_steps)
        model_class = SAC
        if demo_data is not None:
            from demonstrations import DemonstrationReplayBuffer, DemonstrationSAC, imitate_actor
            model_class = DemonstrationSAC
            if demo_data['observations'].shape[1] != env.observation_space.shape[0]:
                raise ValueError('Demonstration observation dimension differs from the training environment')
            if demo_meta['episode_seconds'] != cfg.episode_length_s or demo_meta['step_dt'] != env.unwrapped.step_dt:
                raise ValueError('Demonstration clock differs from training: recollect with the normal lift timeout')
            settings.update(learning_starts=0, ent_coef='auto_0.001',
                            replay_buffer_class=DemonstrationReplayBuffer,
                            replay_buffer_kwargs=dict(handle_timeout_termination=True,
                                demo_path=str(Path(demo_path).resolve()),
                                demo_fraction=args.demo_fraction))
        if args.resume:
            model = model_class.load(args.resume, env=env, device=args.device, tensorboard_log=str(log / 'tensorboard'))
            model.load_replay_buffer(replay_path)
        else:
            if args.initialize_from:
                # Collect from the pretrained actor immediately; previous-phase transitions are discarded.
                settings['learning_starts'] = 0
            model = model_class('MlpPolicy', env, seed=args.seed, device=args.device,
                        tensorboard_log=str(log / 'tensorboard'), **settings)
            if args.initialize_from:
                old = SAC.load(args.initialize_from, device=args.device)
                model.policy.load_state_dict(old.policy.state_dict())
                del old
        if demo_data is not None:
            model._demo_data = demo_data
            if not args.resume:
                model.bc_weight = args.bc_weight
        serial_settings = {k: (v.__name__ if isinstance(v, type) else v) for k, v in settings.items()}
        manifest = dict(algorithm='SAC', schema=SCHEMA, task=TASK, phase=args.phase, seed=args.seed,
                        source=args.source, num_envs=args.num_envs, versions=versions(), fingerprint=fingerprint(),
                        observation_dim=env.observation_space.shape[0], action_dim=14,
                        settings=serial_settings if not args.resume else meta['settings'],
                        parent_checkpoint=str(prior.resolve()) if prior else None,
                        assets=asset_hashes(), evaluation_required=True)
        if demo_data is not None:
            manifest['demonstrations'] = dict(path=str(Path(demo_path).resolve()),
                sha256=demo_meta['sha256'], transitions=demo_meta['transitions'],
                bc_steps=args.bc_steps if not args.resume else meta['demonstrations']['bc_steps'],
                bc_weight=model.bc_weight, demo_fraction=model.replay_buffer.demo_fraction)
            shutil.copy2(Path(demo_path).with_suffix('.json'), log / 'demonstrations.json')
            shutil.copy2(REPO / 'workflows/robotic_surgery/scripts/simulation/scripts/environments/diagnose_peg_grasp.py',
                         log / 'collector.py')
        from robotic.surgery.tasks.surgical.peg_transfer import sac_geometry as source_geometry
        manifest['source_positions_xy'] = {name: list(map(float, xy)) for name, xy in
                                            zip(source_geometry.SOURCE_LABELS, source_geometry.SOURCE_POSTS)}
        (log / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        dump_yaml(str(log / 'env.yaml'), cfg)
        for source in source_files():
            dest = log / 'sources' / source.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
        (log / 'git.diff').write_text(subprocess.run(['git', 'diff', '--', 'workflows', 'PEG_TRANSFER.md'],
                                                    cwd=REPO, capture_output=True, text=True, check=True).stdout)
        (log / 'git.status').write_text(subprocess.run(['git', 'status', '--short'], cwd=REPO,
                                                      capture_output=True, text=True, check=True).stdout)

        class Checkpoints(BaseCallback):
            last_saved = model.num_timesteps
            def _on_step(self):
                for info in self.locals['infos']:
                    if 'transfer' in info:
                        for name, achieved in info['transfer']['metrics'].items():
                            self.logger.record_mean('transfer/' + name, float(achieved))
                        with (log / 'episodes.jsonl').open('a') as f:
                            f.write(json.dumps(dict(timesteps=self.num_timesteps, **info['transfer'],
                                                    **info['episode'])) + '\n')
                if self.num_timesteps - self.last_saved >= args.save_every:
                    save_checkpoint(self.model, log / f'sac_{self.num_timesteps}.zip', manifest)
                    self.last_saved = self.num_timesteps
                return True

        print(f'[SAC] Run: {log}\n[SAC] Model settings: {manifest}', flush=True)
        if demo_data is not None and not args.resume:
            progress = []
            for start in range(0, args.bc_steps, 100):
                loss = imitate_actor(model, demo_data, min(100, args.bc_steps - start), weight=args.bc_weight)
                progress.append(dict(updates=min(start + 100, args.bc_steps), loss=loss))
                if (start + 100) % 500 == 0 or start + 100 >= args.bc_steps:
                    print(f'[BC] {progress[-1]}', flush=True)
            (log / 'imitation.json').write_text(json.dumps(progress, indent=2) + '\n')
            save_checkpoint(model, log / 'pretrained.zip', manifest)
            print(f'[SAC] Actor after imitation: {log / "pretrained.zip"}; requires simulator evaluation', flush=True)
        if not args.imitation_only:
            try:
                model.learn(total_timesteps=args.steps, callback=Checkpoints(), reset_num_timesteps=not bool(args.resume))
            except KeyboardInterrupt:
                print('[SAC] Interrupted: saving current model and replay', flush=True)
        save_checkpoint(model, log / 'final.zip', manifest, replay=True)
        print(f'[SAC] Checkpoint: {log / "final.zip"}', flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
