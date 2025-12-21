import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from tqdm import tqdm

from environment.env import EasyTwoPlayerEnv
from agent.meta.wrapper import RL2Wrapper
from agent.interface import AGENTS
from agent.interface import OPPONENT

def makeInferEnv(n_episodes):
    return DummyVecEnv([lambda: RL2Wrapper(episodes_per_task=n_episodes, mode='inference')])

def makeEasyEnv(opponent_agent):
    return EasyTwoPlayerEnv(opponent_agent=opponent_agent)

def inference(model_path='models/2M', n_episodes=10000, agent='agent'):
    if agent == 'agent':
        env = makeInferEnv(n_episodes)
        model = RecurrentPPO.load(model_path, env=env)
    else:
        env = makeEasyEnv(OPPONENT)
        model = AGENTS[agent]

    episode_rewards = []
    batch_winrates = []

    wins = 0
    draws = 0
    batch_wins = 0
    batch_draws = 0

    if agent == 'agent':
        obs = env.reset()
        for ep in tqdm(range(n_episodes), desc="Evaluation"):
            ep_reward = 0.0
            preflop = True

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, _, info = env.step(action)
                ep_reward += reward[0]

                if info[0].get("hand_done", False):
                    break

            episode_rewards.append(ep_reward)
            if ep_reward > 0:
                wins += 1
                batch_wins += 1
            elif ep_reward == 0:
                draws += 1
                batch_draws += 1

            if (ep + 1) % 100 == 0:
                batch_winrates.append(batch_wins / 100)
                batch_wins = 0
                batch_draws = 0

        episode_rewards = np.array(episode_rewards)
        batch_winrates = np.array(batch_winrates)

        width = 25
        print(
            "\n" + "=" * width + "\n"
            f"| {'Inference Summary':^{width-4}} |\n"
            + "-" * width + "\n"
            f"| {'Win Rate':<10}| {wins / n_episodes:<10.3f}|\n"
            f"| {'Draw Rate':<10}| {draws / n_episodes:<10.3f}|\n"
            f"| {'Mean Rwd':<10}| {episode_rewards.mean():<10.2f}|\n"
            + "=" * width
        )

    else:
        vpip = 0
        call = 0
        bet = 0

        for ep in tqdm(range(n_episodes), desc="Evaluation"):
            state = env.reset()
            ep_reward = 0.0
            preflop = True

            while True:
                action = model.step(state)

                legal_actions = list(state["legal_actions"].keys())
                if action not in legal_actions:
                    action = int(np.random.choice(legal_actions))

                state, reward, done, info = env.step(action)
                ep_reward += reward

                if not info.get("win", False):
                    if action == 0:
                        call += 1
                    if action == 1:
                        bet += 1
                    if preflop and (action == 0 or action == 1):
                        vpip += 1
                        preflop = False

                if done:
                    break

            episode_rewards.append(ep_reward)
            if ep_reward > 0:
                wins += 1
                batch_wins += 1
            elif ep_reward == 0:
                draws += 1
                batch_draws += 1

            if (ep + 1) % 100 == 0:
                batch_winrates.append(batch_wins / 100)
                batch_wins = 0
                batch_draws = 0

        episode_rewards = np.array(episode_rewards)
        batch_winrates = np.array(batch_winrates)

        vpip = vpip / n_episodes
        agg = bet / (bet + call) if (bet + call) > 0 else 0.0

        width = 25
        print(
            "\n" + "=" * width + "\n"
            f"| {'Inference Summary':^{width-4}} |\n"
            + "-" * width + "\n"
            f"| {'Win Rate':<10}| {wins / n_episodes:<10.3f}|\n"
            f"| {'Draw Rate':<10}| {draws / n_episodes:<10.3f}|\n"
            f"| {'Mean Rwd':<10}| {episode_rewards.mean():<10.2f}|\n"
            f"| {'VPIP':<10}| {vpip:<10.2f}|\n"
            f"| {'Agg':<10}| {agg:<10.2f}|\n"
            + "=" * width
        )

    return episode_rewards, batch_winrates

if __name__ == "__main__":
    episode_rewards, batch_winrates = inference(
        n_episodes=100_000,
        agent='passive',
    )

    '''episodes = np.arange(1, len(episode_rewards) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(episodes, episode_rewards, color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(100, len(batch_winrates) * 100 + 1, 100), batch_winrates, color='green')
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Win Rate vs Episode")

    plt.show()'''
