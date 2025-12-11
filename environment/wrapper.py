import numpy as np
import gym

from environment.env import HoldemTwoPlayerEnv
from agent.interface import OpponentSampler

class RL2Wrapper(gym.Env):
    def __init__(self, episodes_per_task=5):
        self.task = OpponentSampler()
        self.episodes_per_task = episodes_per_task
        self.resetTask()

    def resetTask(self):
        self.episode = 0
        self.env = HoldemTwoPlayerEnv(opponent_agent=self.task.sample())
        obs = self.env.reset()
        self.reward = 0
        return self.augment(obs)

    def reset(self):
        if self.current_episode >= self.episodes_per_task:
            return self.resetTask()

        self.current_episode += 1
        obs = self.env.reset()
        return self.augment(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        aug_obs = self.augment(obs, reward)
        return aug_obs, reward, done, info

    def augment(self, obs, reward=0):
        return np.concatenate([obs, reward])
