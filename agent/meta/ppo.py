from stable_baselines3 import PPO

from lstm import RL2LstmPolicy

class RL2PPO(PPO):
    def __init__(self, env, **kwargs):
        super().__init__(
            policy=RL2LstmPolicy,
            env=env,
            verbose=1,
            n_steps=2048,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=3e-4,
            policy_kwargs=dict(lstm_hidden_size=128),
            **kwargs
        )

    def reset_policy_state(self):
        self.policy.reset_lstm()

    def collect_rollouts(self, env, callback, rollout_buffer, n_steps):
        if hasattr(env, "is_task_reset") and env.is_task_reset:
            self.policy.reset_lstm()
        return super().collect_rollouts(env, callback, rollout_buffer, n_steps)
