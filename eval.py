import numpy as np
import matplotlib.pyplot as plt
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from tqdm import tqdm

from environment.env import EasyTwoPlayerEnv
from agent.meta.wrapper import RL2Wrapper
from agent.interface import AGENTS, agent_sample

MODELS = ["discrete", "continuous", "mix"]

def makeInferEnv(n_episodes, opponent):
    return DummyVecEnv([lambda: RL2Wrapper(episodes_per_task=n_episodes, opponent=opponent, mode='inference')])

def makeEasyEnv(opponent):
    if opponent in AGENTS:
        return EasyTwoPlayerEnv(opponent_agent=AGENTS[opponent])
    else:
        return EasyTwoPlayerEnv(opponent_agent=agent_sample())

def inference(model_path='models/discrete', n_episodes=10000, agent='meta', opponent='baseline'):
    if agent in MODELS:
        env = makeInferEnv(n_episodes, opponent)
        model = RecurrentPPO.load(model_path, env=env)
    else:
        env = makeEasyEnv(opponent)
        if opponent in AGENTS:
            model = AGENTS[agent]
        else:
            model = agent_sample()

    episode_rewards = []
    batch_winrates = []

    wins = 0
    draws = 0
    batch_wins = 0
    batch_draws = 0

    if agent in MODELS:
        if opponent != 'human':
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
            counts = 0
            obs = env.reset()
            try:
                while True:
                    ep_reward = 0.0
                    preflop = True

                    while True:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, _, info = env.step(action)
                        ep_reward += reward[0]

                        if info[0].get("hand_done", False):
                            break

                    counts += 1
                    width = 25
                    print(
                        "\n" + "=" * width + "\n"
                        f"| {'Results':^{width-4}} |\n"
                        + "-" * width + "\n"
                        f"| {'Reward':<10}| {-ep_reward:<10.3f}|\n"
                        f"| {'Episode':<10}| {counts:<10}|\n"
                        + "=" * width
                    )

                    episode_rewards.append(ep_reward)
                    if ep_reward > 0:
                        wins += 1
                        batch_wins += 1
                    elif ep_reward == 0:
                        draws += 1
                        batch_draws += 1

                    if (counts + 1) % 100 == 0:
                        batch_winrates.append(batch_wins / 100)
                        batch_wins = 0
                        batch_draws = 0

            except:
                episode_rewards = np.array(episode_rewards)
                batch_winrates = np.array(batch_winrates)

                width = 25
                print(
                    "\n" + "=" * width + "\n"
                    f"| {'Inference Summary':^{width-4}} |\n"
                    + "-" * width + "\n"
                    f"| {'Win Rate':<10}| {wins / counts:<10.3f}|\n"
                    f"| {'Draw Rate':<10}| {draws / counts:<10.3f}|\n"
                    f"| {'Mean Rwd':<10}| {episode_rewards.mean():<10.2f}|\n"
                    + "=" * width
                )

    elif agent == 'human':
        vpip = 0
        call = 0
        bet = 0
        counts = 0
        try:
            while True:
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

                width = 25
                print(
                    "\n" + "=" * width + "\n"
                    f"| {'Your Reward':^{width-4}} |\n"
                    + "-" * width + "\n"
                    f"| {ep_reward:^{width-4}.3f} |\n"
                    + "=" * width
                )

                episode_rewards.append(ep_reward)
                if ep_reward > 0:
                    wins += 1
                    batch_wins += 1
                elif ep_reward == 0:
                    draws += 1
                    batch_draws += 1

                if (counts + 1) % 100 == 0:
                    batch_winrates.append(batch_wins / 100)
                    batch_wins = 0
                    batch_draws = 0

                counts += 1

        except:
            episode_rewards = np.array(episode_rewards)
            batch_winrates = np.array(batch_winrates)

            vpip = vpip / counts
            agg = bet / (bet + call) if (bet + call) > 0 else 0.0

            width = 25
            print(
                "\n" + "=" * width + "\n"
                f"| {'Inference Summary':^{width-4}} |\n"
                + "-" * width + "\n"
                f"| {'Win Rate':<10}| {wins / counts:<10.3f}|\n"
                f"| {'Draw Rate':<10}| {draws / counts:<10.3f}|\n"
                f"| {'Mean Rwd':<10}| {episode_rewards.mean():<10.2f}|\n"
                f"| {'VPIP':<10}| {vpip:<10.2f}|\n"
                f"| {'Agg':<10}| {agg:<10.2f}|\n"
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference script for RL2 Hold'em agents"
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="meta",
        choices=MODELS + list(AGENTS.keys()) + ["param"],
        help="Which agent to evaluate"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="baseline",
        choices=list(AGENTS.keys()) + ["param"],
        help="Which agent as the opponent"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Evaluate with less episodes"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot reward and win rate curves"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    episode_rewards, batch_winrates = inference(
        n_episodes=100_000 if not args.fast else 10_000,
        model_path=f'models/{args.agent}',
        agent=args.agent,
        opponent = args.opponent,
    )

    if args.plot:
        episodes = np.arange(1, len(episode_rewards) + 1)

        plt.figure(figsize=(6, 4))
        plt.plot(episodes, episode_rewards, color='orange')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward vs Episode")

        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, len(batch_winrates) + 1, 1), batch_winrates, color='green')
        plt.xlabel("Batch")
        plt.ylabel("Win Rate")
        plt.title("Win Rate vs Episode")

        plt.show()
