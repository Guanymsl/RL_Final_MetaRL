import numpy as np
import gym

from environment.env import HoldemTwoPlayerEnv

class OpponentSampler:
    def __init__(self, opponents):
        self.opponents = opponents

    def sample(self):
        return np.random.choice(self.opponents)

class RL2Wrapper(gym.ObservationWrapper):
    def __init__(self, env, latent_dim=259):
        super().__init__(env)

        self.latent_dim = latent_dim
        self.aug_dim = self.latent_dim + 1

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.aug_dim,),
            dtype=np.float32
        )

        self.last_reward = np.zeros(1, dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        self.last_reward[0] = 0.0
        return self.augment(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last_reward[0] = reward
        return self.augment(obs), reward, done, info

    def augment(self, latent):
        return np.concatenate([latent, self.last_reward], axis=0)

def makeMetaEnv(opponent_sampler: OpponentSampler) -> gym.Env:
    opponent = opponent_sampler.sample()
    base_env = HoldemTwoPlayerEnv(opponent_agent=opponent)
    return RL2Wrapper(base_env)
