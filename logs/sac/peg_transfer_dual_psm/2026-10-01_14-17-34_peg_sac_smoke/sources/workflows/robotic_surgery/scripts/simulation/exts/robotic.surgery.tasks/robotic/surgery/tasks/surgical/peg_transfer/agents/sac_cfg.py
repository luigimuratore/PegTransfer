"""Native Stable-Baselines3 2.9 SAC settings (no custom SAC implementation)."""

SAC_SETTINGS = dict(
    learning_rate=3e-4,
    buffer_size=500_000,  # total transitions, not 500k per parallel environment
    learning_starts=10_000,
    batch_size=256,
    tau=0.005,
    gamma=0.999,  # 1000 control steps in a full episode: retain terminal credit
    train_freq=1,
    gradient_steps=4,  # four minibatches for each vector step; tune against num_envs
    ent_coef='auto_0.1',
    target_entropy=-14.0,
    policy_kwargs=dict(net_arch=[256, 256], n_critics=2),
    replay_buffer_kwargs=dict(handle_timeout_termination=True),
    optimize_memory_usage=False,
    verbose=1,
)
