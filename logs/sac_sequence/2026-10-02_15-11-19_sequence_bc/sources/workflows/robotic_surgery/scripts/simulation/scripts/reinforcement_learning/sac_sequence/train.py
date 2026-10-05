"""Train sequence-v1 SAC or pretrain its goal-conditioned actor only."""
from datetime import datetime
import json
import shutil
import subprocess
import traceback
from common import REPO, SCHEMA, TASK, assets, checkpoint, cli, fingerprint, make_env, save, sources, versions

def main():
    args = cli('train')
    prior = checkpoint(args.initialize_from) if args.initialize_from else None
    if prior and prior['source'] != args.source:
        raise ValueError('Initial curriculum keeps the same fixed source and R2 target')
    if args.guidance is None:
        args.guidance = prior['guidance'] if prior else 1.
    from model import AnchoredSAC, DemoReplay, load_demos, pretrain
    from policy import GoalPolicy
    data, demo = load_demos(args.demonstrations)
    if demo['source'] != args.source or demo['episode_seconds'] != args.episode_seconds:
        raise ValueError('Demo source/episode clock differs from training')
    if versions()['stable-baselines3'] != '2.9.0':
        raise ValueError('Anchored SAC was verified against local SB3 2.9.0; recheck API before changing versions')
    from isaaclab.app import AppLauncher
    app, env = None, None
    try:
        print('[SEQUENCE SAC] Starting Isaac Sim',flush=True)
        app = AppLauncher(args).app
        env, cfg = make_env(args)
        if data['observations'].shape[1] != env.observation_space.shape[0]:
            raise ValueError('Demo observation dimensions differ from the sequence environment')
        from stable_baselines3.common.callbacks import BaseCallback
        from isaaclab.utils.io import dump_yaml
        from robotic.surgery.tasks.surgical.peg_transfer_sequence import geometry
        run = REPO/'logs/sac_sequence'/f'{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.run_name}'
        run.mkdir(parents=True,exist_ok=False)
        settings = dict(learning_rate=args.learning_rate, buffer_size=args.buffer_size,
            learning_starts=0, batch_size=args.batch_size, tau=.005, gamma=cfg.sequence_gamma,
            train_freq=1, gradient_steps=args.gradient_steps, ent_coef=args.ent_coef,
            target_entropy=-4., replay_buffer_class=DemoReplay,
            replay_buffer_kwargs=dict(demo_path=str(args.demonstrations.resolve()), demo_fraction=args.demo_fraction,
                                      handle_timeout_termination=True),
            policy_kwargs=dict(net_arch=[256,256], n_critics=2, share_features_extractor=False), verbose=1)
        model = AnchoredSAC(GoalPolicy,env,seed=args.seed,device=args.device,
                            tensorboard_log=str(run/'tensorboard'),**settings)
        if args.initialize_from:
            previous = AnchoredSAC.load(args.initialize_from,device=args.device)
            model.policy.load_state_dict(previous.policy.state_dict())
            del previous  # Fresh replay: no old curriculum terminal boundaries are reused.
        model.bc_weight, model._demo_data = args.bc_weight, data
        manifest = dict(schema=SCHEMA,task=TASK,algorithm='SAC+joint-BC',fingerprint=fingerprint(),assets=assets(),
            source=args.source,target='R2',skill=args.skill,guidance=args.guidance,versions=versions(),seed=args.seed,
            num_envs=args.num_envs,observation_dim=env.observation_space.shape[0],actor_features=29,action_dim=4,
            receiver_grasp_offset_m=list(geometry.GRASP2),receiver_approach='positive_y_edge',
            parent=str(args.initialize_from.resolve()) if args.initialize_from else None,
            demonstrations=dict(path=str(args.demonstrations.resolve()),sha256=demo['sha256'],transitions=demo['transitions']),
            settings={key: value.__name__ if isinstance(value,type) else value for key,value in settings.items()},
            bc_weight=args.bc_weight,bc_steps=args.bc_steps,imitation_only=args.imitation_only,
            episode_seconds=args.episode_seconds,stage_budget=args.stage_budget,
            evaluation_required=True,assisted_grasp=True,physical_grasp_verified=False)
        (run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        dump_yaml(str(run/'env.yaml'),cfg)
        for source in sources():
            dest = run/'sources'/source.relative_to(REPO)
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,dest)
        (run/'git.diff').write_text(subprocess.run(['git','diff','--','workflows','PEG_TRANSFER.md'],cwd=REPO,
                                                  capture_output=True,text=True,check=True).stdout)
        (run/'git.status').write_text(subprocess.run(['git','status','--short'],cwd=REPO,
                                                    capture_output=True,text=True,check=True).stdout)
        print(f'[SEQUENCE SAC] Run: {run}',flush=True)
        pretrain(model,data,args.bc_steps,args.batch_size)
        save(model,run/'pretrained.zip',manifest)
        class Log(BaseCallback):
            last_saved = 0
            def _on_step(self):
                for info in self.locals['infos']:
                    if 'transfer' in info:
                        with (run/'episodes.jsonl').open('a') as file:
                            file.write(json.dumps(dict(timesteps=self.num_timesteps,**info['transfer'],**info['episode']))+'\n')
                        for key, value in info['transfer']['metrics'].items():
                            self.logger.record_mean('transfer/'+key,float(value))
                        self.logger.record_mean('transfer/skill_success',float(info['transfer']['phase_success']))
                if self.num_timesteps-self.last_saved >= args.save_every:
                    save(self.model,run/f'sac_{self.num_timesteps}.zip',manifest)
                    self.last_saved = self.num_timesteps
                return True
        if not args.imitation_only:
            try:
                model.learn(args.steps,callback=Log())
            except KeyboardInterrupt:
                print('[SEQUENCE SAC] Interrupted; saving current state',flush=True)
        save(model,run/'final.zip',manifest,replay=not args.imitation_only)
        print(f'[SEQUENCE SAC] Checkpoint: {run / "final.zip"}',flush=True)
    except Exception:
        traceback.print_exc()  # Print before Kit close can exit the interpreter.
        raise
    finally:
        if env is not None: env.close()
        if app is not None: app.close()

if __name__ == '__main__':
    main()
