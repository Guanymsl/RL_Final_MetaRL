from stable_baselines3 import PPO

class RL2PPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_policy_state(self):
        self.policy.reset_lstm()

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        if hasattr(env, "envs"):
            for idx, subenv in enumerate(env.envs):
                if hasattr(subenv, "is_task_reset") and subenv.is_task_reset:
                    self.policy.reset_lstm(env_idx=idx)

        return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
