import numpy as np
import gymnasium as gym

from environment.env import HoldemTwoPlayerEnv
from agent.interface import OpponentSampler
from preprocess.param import AE_LATENT_DIM

class RL2Wrapper(gym.Env):
    def __init__(self, episodes_per_task=100, mode='train'):
        super().__init__()

        self.task = OpponentSampler(mode=mode)
        self.episodes_per_task = episodes_per_task
        self.mode = mode

        self.last_obs = None
        self.just_reset = False

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
        self.env = HoldemTwoPlayerEnv(opponent_agent=self.task.sample(), mode=self.mode)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.mode == 'train':
            self._reset_task()

        obs = self.env.reset()
        return self.augment(obs), {}

    def step(self, action):
        if self.just_reset:
            self.just_reset = False
            return self.augment(self.last_obs), 0.0, False, False, {"just": True}

        obs, reward, terminated, truncated, info = self.env.step(action)
        aug_obs = self.augment(obs, reward)

        if self.mode == 'train':
            real_done = False

            if terminated or truncated:
                self.episode += 1
                info = {"hand_done": True, "reward": reward, **info}

                if self.episode >= self.episodes_per_task:
                    real_done = True
                else:
                    self.last_obs = self.env.reset()
                    self.just_reset = True

            return aug_obs, reward, real_done, False, info

        else:
            if terminated or truncated:
                self.last_obs = self.env.reset()
                self.just_reset = True

                info = {"hand_done": True, **info}

            return aug_obs, reward, False, False, info

    def augment(self, obs, reward=0):
        return np.concatenate([obs, [reward]]).astype(np.float32)
