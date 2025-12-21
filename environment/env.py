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

    def _get_obs(self, player):
        state = self.env.get_state(player)
        return self.preprocessor.encode(state["obs"].astype(np.float32))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        state, player = self.env.reset()
        self.current_player = player

        while self.current_player == 1 and not self.env.is_over():
            state, self.current_player = self.env.step(
                self.opponent.step(state)
            )

        self.start_obs = self._get_obs(self.current_player)
        self.prev_chips = float(state["raw_obs"]["all_chips"][0])

        return self.start_obs

    def step(self, action):
        if self.env.is_over():
            reward = self.env.get_payoffs()[0]
            return self.start_obs, reward, True, False, {"win": True}

        state = self.env.get_state(self.current_player)
        legal_actions = list(state["legal_actions"].keys())
        if action not in legal_actions:
            action = int(np.random.choice(legal_actions))

        state, next_player = self.env.step(action)
        self.current_player = next_player

        if self.env.is_over():
            reward = self.env.get_payoffs()[0]
            return self.preprocessor.encode(state["obs"].astype(np.float32)), reward, True, False, {}

        before = self.prev_chips
        self.prev_chips = float(state["raw_obs"]["all_chips"][0])
        reward = 0.1 * (self.prev_chips - before)

        while self.current_player == 1 and not self.env.is_over():
            state, self.current_player = self.env.step(
                self.opponent.step(state)
            )

        obs = self._get_obs(self.current_player)

        if self.env.is_over():
            reward = self.env.get_payoffs()[0]
            return obs, reward, True, False, {}

        return obs, reward, False, False, {}

class EasyTwoPlayerEnv():
    metadata = {"render.modes": []}

    def __init__(self, opponent_agent, game_name="limit-holdem"):
        self.env = rlcard.make(game_name, config={"allow_step_back": False})
        if hasattr(self.env.game, "allowed_raise_num"):
            self.env.game.allowed_raise_num = 2

        self.opponent = opponent_agent
        self.current_player = None

    def _get_state(self, player):
        state = self.env.get_state(player)
        return state

    def reset(self):
        state, player = self.env.reset()
        self.current_player = player

        while self.current_player == 1 and not self.env.is_over():
            state, self.current_player = self.env.step(
                self.opponent.step(state)
            )

        self.start_state = self._get_state(self.current_player)
        return self._get_state(self.current_player)

    def step(self, action):
        if self.env.is_over():
            return self.start_state, self.env.get_payoffs()[0], True, {"win": True}

        state = self.env.get_state(self.current_player)
        legal_actions = list(state["legal_actions"].keys())
        if action not in legal_actions:
            action = int(np.random.choice(legal_actions))

        state, next_player = self.env.step(action)
        self.current_player = next_player

        if self.env.is_over():
            return state, self.env.get_payoffs()[0], True, {}

        while self.current_player == 1 and not self.env.is_over():
            state, self.current_player = self.env.step(
                self.opponent.step(state)
            )

        state = self._get_state(self.current_player)

        if self.env.is_over():
            return state, self.env.get_payoffs()[0], True, {}

        return state, 0.0, False, {}
