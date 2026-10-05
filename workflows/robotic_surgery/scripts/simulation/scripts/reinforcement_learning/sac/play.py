"""Graphical deterministic SAC inference in the complete transfer environment."""

from common import make_env, parser, read_checkpoint


def main():
    args = parser('play')
    read_checkpoint(args.checkpoint)
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    env = None
    try:
        from stable_baselines3 import SAC
        env, _ = make_env(args, 'full')
        model = SAC.load(args.checkpoint, env=env, device=args.device)
        obs, step = env.reset(), 0
        while app.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, infos = env.step(action)
            step += 1
            for info in infos:
                if 'transfer' in info:
                    print('[PLAY]', info['transfer'], flush=True)
            if (args.max_steps and step >= args.max_steps) or (args.video and step >= args.video_length):
                break
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
