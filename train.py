from environment.wrapper import RL2Wrapper
from agent.meta.ppo import RL2PPO

from stable_baselines3.common.vec_env import DummyVecEnv

def makeVecEnv():
    return DummyVecEnv([lambda: RL2Wrapper(episodes_per_task=5)])

def main():
    env = makeVecEnv()
    model = RL2PPO(env)
    model.learn(total_timesteps=500_000)
    model.save("metaholdem")

if __name__ == "__main__":
    main()
