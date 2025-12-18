import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from tqdm import tqdm

from environment.wrapper import RL2Wrapper

def makeInferEnv(n_episodes):
    return DummyVecEnv([lambda: RL2Wrapper(episodes_per_task=n_episodes, mode='inference')])

def inference(model_path: str, n_episodes: int = 100):
    env = makeInferEnv(n_episodes)
    model = RecurrentPPO.load(model_path, env=env)

    episode_rewards = []
    wins = 0

    for ep in tqdm(range(n_episodes), desc="Inference"):
        obs = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]

        episode_rewards.append(ep_reward)
        if ep_reward > 0:
            wins += 1

    episode_rewards = np.array(episode_rewards)

    width = 25
    print(
        "\n" + "=" * width + "\n"
        f"| {'Inference Summary':^{width-4}} |\n"
        + "-" * width + "\n"
        f"| {'Win Rate':<10}| {wins / n_episodes:<10.3f}|\n"
        f"| {'Mean Rwd':<10}| {episode_rewards.mean():<10.2f}|\n"
        + "=" * width
    )

    return episode_rewards


if __name__ == "__main__":
    inference(
        model_path="models/metaholdem",
        n_episodes=100000,
    )
