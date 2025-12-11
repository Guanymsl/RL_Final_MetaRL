from environment.env import HoldemTwoPlayerEnv
from environment.wrapper import RL2Wrapper

from agent.meta.ppo import RL2PPO

def makeEnv():
    env = RL2Wrapper(episodes_per_task=5)
    return env

def main():
    env = makeEnv()

    model = RL2PPO(env)
    model.learn(total_timesteps=500_000)
    model.save("rl2_lstm_ppo_holdem")

if __name__ == "__main__":
    main()
