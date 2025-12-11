import numpy as np
import rlcard
import gym
from gym import spaces

from preprocess.preproc import GameStateToTensor

class HoldemTwoPlayerEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, opponent_agent, game_name="limit-holdem"):
        super().__init__()

        self.env = rlcard.make(game_name, config={"allow_step_back": False})
        if hasattr(self.env.game, "allowed_raise_num"):
            self.env.game.allowed_raise_num = 2

        self.opponent = opponent_agent

        obs_dim = self.env.state_shape["obs"][0]
        self.action_dim = self.env.action_num

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.action_dim)

        self.preprocessor = GameStateToTensor(
            latent_card_dim=128,
            latent_action_dim=128,
            stack_dim=3
        )

        self.current_player = None

    def reset(self):
        state, player = self.env.reset()
        self.current_player = player

        while self.current_player == 1 and not self.env.is_over():
            state, self.current_player = self.env.step(
                self.opponent.act(state)
            )

        return self.preprocessor.encode(state["obs"].astype(np.float32))

    def step(self, action):
        state, next_player = self.env.step(action)
        self.current_player = next_player

        if self.env.is_over():
            payoffs = self.env.get_payoffs()
            reward = payoffs[0]
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, reward, True, {}

        while self.current_player == 1 and not self.env.is_over():
            state, self.current_player = self.env.step(
                self.opponent.act(state)
            )

        if self.env.is_over():
            payoffs = self.env.get_payoffs()
            reward = payoffs[0]
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, reward, True, {}

        obs = self.preprocessor.encode(state["obs"].astype(np.float32))
        return obs, 0.0, False, {}
