import numpy as np
import gym

from environment.env import HoldemTwoPlayerEnv
from agent.interface import OpponentSampler

class RL2Wrapper(gym.Env):
    def __init__(self, episodes_per_task=5):
        super().__init__()

        self.task = OpponentSampler()
        self.episodes_per_task = episodes_per_task

        self.resetTask()

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.env.observation_space.shape[0] + 1,),
            dtype=np.float32
        )
        self.action_space = self.env.action_space

    def resetTask(self):
        self.episode = 0
        self.is_task_reset = True

        self.env = HoldemTwoPlayerEnv(opponent_agent=self.task.sample())
        obs = self.env.reset()
        return self.augment(obs)

    def reset(self):
        if self.episode >= self.episodes_per_task:
            return self.resetTask()

        self.episode += 1
        self.is_task_reset = False

        obs = self.env.reset()
        return self.augment(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        aug_obs = self.augment(obs, reward)
        return aug_obs, reward, done, info

    def augment(self, obs, reward=0):
        return np.concatenate([obs, [reward]]).astype(np.float32)
