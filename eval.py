import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from agent.meta.ppo import RL2PPO
from environment.wrapper import RL2Wrapper


def makeInferEnv(n_episodes):
    return DummyVecEnv([lambda: RL2Wrapper(episodes_per_task=n_episodes)])


def inference(model_path: str, n_episodes: int = 100):
    env = makeInferEnv(n_episodes)
    model = RL2PPO.load(model_path, env=env)

    episode_rewards = []
    wins = 0

    for ep in range(n_episodes):
        obs = env.reset()
        model.policy.reset_lstm(env_idx=0)
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]

        episode_rewards.append(ep_reward)
        if ep_reward > 0:
            wins += 1

        print(f"Episode {ep + 1:03d} | reward = {ep_reward:.2f}")

    episode_rewards = np.array(episode_rewards)

    print("\n===== Inference Summary =====")
    print(f"Episodes     : {n_episodes}")
    print(f"Win rate     : {wins / n_episodes:.3f}")
    print(f"Mean reward  : {episode_rewards.mean():.2f}")
    print(f"Std reward   : {episode_rewards.std():.2f}")

    return episode_rewards


if __name__ == "__main__":
    inference(
        model_path="metaholdem",
        n_episodes=200,
    )
