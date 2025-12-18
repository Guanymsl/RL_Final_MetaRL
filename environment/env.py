import numpy as np
import rlcard
import gymnasium as gym

from preprocess.preproc import GameStateToTensor
from preprocess.param import AE_LATENT_DIM

class HoldemTwoPlayerEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, opponent_agent, game_name="limit-holdem"):
        super().__init__()

        self.env = rlcard.make(game_name, config={"allow_step_back": False})
        if hasattr(self.env.game, "allowed_raise_num"):
            self.env.game.allowed_raise_num = 2

        self.opponent = opponent_agent

        obs_dim = self.env.state_shape[0]
        if isinstance(obs_dim, list):
            obs_dim = obs_dim[0]

        self.action_dim = 4

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(self.action_dim)

        self.preprocessor = GameStateToTensor(latent_dim=AE_LATENT_DIM)

        self.current_player = None

        self.prev_chips = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        state, player = self.env.reset()
        self.current_player = player

        while self.current_player == 1 and not self.env.is_over():
            state, self.current_player = self.env.step(
                self.opponent.step(state)
            )

        self.prev_chips = float(state["raw_obs"]["all_chips"][0])
        return self.preprocessor.encode(state["obs"].astype(np.float32))

    def step(self, action):
        state, next_player = self.env.step(action)
        self.current_player = next_player

        before = self.prev_chips
        self.prev_chips = float(state["raw_obs"]["all_chips"][0])
        reward = 0.1 * (self.prev_chips - before)

        if self.env.is_over():
            payoffs = self.env.get_payoffs()
            reward = payoffs[0]
            obs = np.zeros(AE_LATENT_DIM, dtype=np.float32)
            return obs, reward, True, False, {}

        while self.current_player == 1 and not self.env.is_over():
            state, self.current_player = self.env.step(
                self.opponent.step(state)
            )

        if self.env.is_over():
            payoffs = self.env.get_payoffs()
            reward = payoffs[0]
            obs = np.zeros(AE_LATENT_DIM, dtype=np.float32)
            return obs, reward, True, False, {}

        obs = self.preprocessor.encode(state["obs"].astype(np.float32))
        return obs, reward, False, False, {}
