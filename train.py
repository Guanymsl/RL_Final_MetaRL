import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO

from agent.meta.wrapper import RL2Wrapper
from agent.meta.callback import WinRateCallback

def makeVecEnv():
    return DummyVecEnv([lambda: Monitor(RL2Wrapper(episodes_per_task=100, mode='train'))])

def main():
    wandb.init(
        project="meta-holdem",
        name="RL2",
    )

    env = makeVecEnv()
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        verbose=1,
        n_steps=4096,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=1e-4,
        policy_kwargs=dict(
            shared_lstm=False,
            enable_critic_lstm=True,
        ),
    )

    win_rate_callback = WinRateCallback(
        batch_size=1000,
        verbose=1,
    )

    wandb_callback = WandbCallback(
        model_save_path="wandb_models/",
        model_save_freq=100_000,
        verbose=1,
    )

    model.learn(
        total_timesteps=100_000,
        callback=CallbackList([
            win_rate_callback,
            wandb_callback,
        ])
    )

    model.save("models/metaholdem")

if __name__ == "__main__":
    main()
