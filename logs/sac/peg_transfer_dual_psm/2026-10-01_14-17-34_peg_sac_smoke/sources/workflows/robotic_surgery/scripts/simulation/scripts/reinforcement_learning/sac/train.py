"""Train native Stable-Baselines3 SAC; never load an RSL-RL/PPO checkpoint."""

from datetime import datetime
import json
import subprocess
import shutil

from common import REPO, SCHEMA, TASK, asset_hashes, fingerprint, make_env, parser, read_checkpoint, save_checkpoint, source_files, versions


def main():
    args = parser('train')
    prior = args.resume or args.initialize_from
    meta = read_checkpoint(prior) if prior else None  # reject PPO before launching Sim
    if args.resume and (meta['phase'] != args.phase or meta['num_envs'] != args.num_envs):
        raise ValueError('Resume requires same phase and num_envs; use initialize_from for a new phase')
    replay_path = prior.with_name(prior.stem + '_replay.pkl') if args.resume else None
    if replay_path and not replay_path.is_file():
        raise FileNotFoundError(f'Resume requires replay buffer: {replay_path}; use a final checkpoint')
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
        if args.resume:
            model = SAC.load(args.resume, env=env, device=args.device, tensorboard_log=str(log / 'tensorboard'))
            model.load_replay_buffer(replay_path)
        else:
            if args.initialize_from:
                # Collect from the pretrained actor immediately; previous-phase transitions are discarded.
                settings['learning_starts'] = 0
            model = SAC('MlpPolicy', env, seed=args.seed, device=args.device,
                        tensorboard_log=str(log / 'tensorboard'), **settings)
            if args.initialize_from:
                old = SAC.load(args.initialize_from, device=args.device)
                model.policy.load_state_dict(old.policy.state_dict())
                del old
        manifest = dict(algorithm='SAC', schema=SCHEMA, task=TASK, phase=args.phase, seed=args.seed,
                        source=args.source, num_envs=args.num_envs, versions=versions(), fingerprint=fingerprint(),
                        observation_dim=env.observation_space.shape[0], action_dim=14,
                        settings=settings if not args.resume else meta['settings'],
                        parent_checkpoint=str(prior.resolve()) if prior else None,
                        assets=asset_hashes(), evaluation_required=True)
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
