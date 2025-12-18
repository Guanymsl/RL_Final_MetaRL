import numpy as np
import gymnasium as gym

from environment.env import HoldemTwoPlayerEnv
from agent.interface import OpponentSampler
from environment.param import AE_LATENT_DIM

class RL2Wrapper(gym.Env):
    def __init__(self, episodes_per_task=10, mode='train'):
        super().__init__()

        self.task = OpponentSampler(mode=mode)
        self.episodes_per_task = episodes_per_task
        self.mode = mode

        self._reset_task()

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(AE_LATENT_DIM + 1,),
            dtype=np.float32
        )
        self.action_space = self.env.action_space

    def _reset_task(self):
        self.episode = 0
        self.env = HoldemTwoPlayerEnv(opponent_agent=self.task.sample())

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.mode == 'train':
            self._reset_task()

        obs = self.env.reset()
        return self.augment(obs), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        aug_obs = self.augment(obs, reward)
        self.episode += 1

        if self.mode == 'train':
            real_done = False

            if terminated or truncated:
                if self.episode >= self.episodes_per_task:
                    real_done = True
                else:
                    obs = self.env.reset()

            return aug_obs, reward, real_done, False, info

        else:
            return aug_obs, reward, terminated, truncated, info

    def augment(self, obs, reward=0):
        return np.concatenate([obs, [reward]]).astype(np.float32)
