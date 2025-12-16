from environment.wrapper import RL2Wrapper
from agent.meta.ppo import RL2PPO

from stable_baselines3.common.vec_env import DummyVecEnv
from agent.meta.lstm import RL2LstmPolicy
from environment.param import LSTM_LATENT_DIM

def makeVecEnv():
    return DummyVecEnv([lambda: RL2Wrapper(episodes_per_task=5)])

def main():
    env = makeVecEnv()

    model = RL2PPO(
        policy=RL2LstmPolicy,
        env=env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        policy_kwargs=dict(lstm_hidden_size=LSTM_LATENT_DIM),
    )

    model.learn(total_timesteps=10000_000)
    model.save("models/metaholdem")

if __name__ == "__main__":
    main()
