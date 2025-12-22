import numpy as np
import rlcard
import gym
from gym import spaces

from wrappers.pt_wrapper import NeuralNetworkAgent
from wrappers.zip_wrapper import PPOAgent


class HoldemTwoPlayerEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self, base_model_path: str, game_name="limit-holdem", aggressive=0.0, tight=0.0
    ):
        super().__init__()

        self.env = rlcard.make(game_name, config={"allow_step_back": False})
        if hasattr(self.env.game, "allowed_raise_num"):
            self.env.game.allowed_raise_num = 2

        if base_model_path.endswith(".pt"):
            self.opponent = NeuralNetworkAgent(base_model_path)
        else:
            self.opponent = PPOAgent(base_model_path)

        obs_dim = self.env.state_shape[0]
        if isinstance(obs_dim, list):
            obs_dim = obs_dim[0]
        # temporary fix
        self.action_dim = 4

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.action_dim)

        self.current_player = None
        self.aggressive = aggressive
        self.tight = tight
        self.raise_count = 0

    def reset(self):
        state, player = self.env.reset()
        self.raise_count = 0
        self.current_player = player

        while self.current_player == 1 and not self.env.is_over():
            state, self.current_player = self.env.step(self.opponent.step(state))

        return state["obs"].astype(np.float32)
        # return self.preprocessor.encode(state["obs"].astype(np.float32))

    def step(self, action):
        state, next_player = self.env.step(action)
        self.current_player = next_player

        if action in [2, 3]:
            self.raise_count += 1

        # if action in [2, 3]:
        #     reward += self.aggressive * 5
        #
        # if action == 0:
        #     reward += self.tight * 5

        # print("shaping reward: ", reward)

        if self.env.is_over():
            payoffs = self.env.get_payoffs()
            reward = payoffs[0]
            reward += self.aggressive * self.raise_count
            if action == 0 and self.env.game.round == 0:
                reward += self.tight
            # print("payoff: ", payoffs[0])
            obs = state["obs"]
            return obs, reward, True, {}

        while self.current_player == 1 and not self.env.is_over():
            state, self.current_player = self.env.step(self.opponent.step(state))

        if self.env.is_over():
            payoffs = self.env.get_payoffs()
            reward = payoffs[0]
            reward += self.aggressive * self.raise_count
            if action == 0 and self.env.game.round == 0:
                reward += self.tight
            # print("payoff: ", payoffs[0])
            obs = state["obs"]
            return obs, reward, True, {}

        obs = state["obs"]
        return obs, 0.0, False, {}
