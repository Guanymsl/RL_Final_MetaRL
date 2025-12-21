import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from environment.env import EasyTwoPlayerEnv
from agent.interface import AGENTS, agent_sample

def makeEasyEnv(opponent):
    return EasyTwoPlayerEnv(opponent_agent=opponent)

def inference(n_episodes=10000):
    env = makeEasyEnv(AGENTS['baseline'])
    model = agent_sample()

    vpip = 0
    call = 0
    bet = 0

    for ep in range(n_episodes):
        state = env.reset()
        preflop = True

        while True:
            action = model.step(state)

            legal_actions = list(state["legal_actions"].keys())
            if action not in legal_actions:
                action = int(np.random.choice(legal_actions))

            state, reward, done, info = env.step(action)

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

    vpip = vpip / n_episodes
    agg = bet / (bet + call) if (bet + call) > 0 else 0.0

    width = 25
    print(
        "\n" + "=" * width + "\n"
        f"| {'Inference Summary':^{width-4}} |\n"
        + "-" * width + "\n"
        f"| {'VPIP':<10}| {vpip:<10.2f}|\n"
        f"| {'Agg':<10}| {agg:<10.2f}|\n"
        + "=" * width
    )

    return vpip, agg

if __name__ == "__main__":
    quad = []
    for sp in tqdm(range(100), desc="Sample"):
        vpip, agg = inference(n_episodes=10_000)
        quad.append((1 - vpip, agg))

    quad = np.array(quad)
    x = quad[:, 0]
    y = quad[:, 1]

    plt.figure(figsize=(7, 7))

    plt.scatter(x, y, alpha=0.8, color='orange')

    x_mid = 0.33
    y_mid = 0.5
    plt.axvline(x_mid, linestyle="--", color='black')
    plt.axhline(y_mid, linestyle="--", color='black')

    plt.text(0.75, 0.75, "Tight / Aggressive", ha="center")
    plt.text(0.25, 0.75, "Loose / Aggressive", ha="center")
    plt.text(0.25, 0.25, "Loose / Passive", ha="center")
    plt.text(0.75, 0.25, "Tight / Passive", ha="center")

    plt.xlabel("Tightness (1 - VPIP)")
    plt.ylabel("Aggression (Agg)")
    plt.title("Style Quadrant")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)

    plt.show()
