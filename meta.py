import numpy as np
from typing import Callable, List, Optional

import gym
from gym import spaces

import torch
import torch.nn as nn

import pyspiel
from open_spiel.python import rl_environment

from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent import MlpLstmPolicy  # 預設是 LSTM，之後可以自訂成 GRU
from stable_baselines3.common.vec_env import DummyVecEnv

# ============================================================
# 1. AE Encoder Stub（你自己換成真正的 AutoEncoder Encoder）
# ============================================================

class PokerAEEncoder(nn.Module):
    """
    假設你已經有一個 train 好的 AE，
    這裡只放 encoder 部分，輸入 info_state 吐出 latent。
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def load_pretrained_ae(input_dim: int, latent_dim: int, ckpt_path: Optional[str] = None) -> PokerAEEncoder:
    ae = PokerAEEncoder(input_dim=input_dim, latent_dim=latent_dim)
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu")
        ae.load_state_dict(state)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    return ae

# ============================================================
# 2. Deep CFR 對手 Wrapper（interface）
# ============================================================

class DeepCFRAgentWrapper:
    """
    包一層，把你的 Deep CFR policy 變成一個
    opponent: time_step -> action (int)
    的 callable。
    """
    def __init__(self, policy):
        """
        policy 可以是你自己實作的 OpenSpiel Deep CFR agent,
        只要有 .step(time_step) -> 有 .action 屬性即可。
        """
        self.policy = policy

    def __call__(self, time_step) -> int:
        # OpenSpiel 的 agent interface 通常是 .step(time_step) 回傳 object，有 .action
        out = self.policy.step(time_step)
        return int(out.action)

# ============================================================
# 3. OpenSpiel Texas Hold'em 1v1 環境（player 0 = 我方）
# ============================================================

class OpenSpielPoker1v1Env(gym.Env):
    """
    單一 RL agent（player 0）對上一個 Deep CFR 對手（player 1）。
    - observation: AE latent of info_state (還沒加 RL^2 的 (a,r,done))
    - action: poker 行為（假設 universal_poker 已設定成離散動作）
    """
    metadata = {"render.modes": []}

    def __init__(
        self,
        game_string: str,
        opponent_policies: List[Callable],
        ae_encoder: PokerAEEncoder,
        max_episode_length: int = 200,
        player_id: int = 0,
    ):
        super().__init__()

        self.game_string = game_string
        self.game = pyspiel.load_game(game_string)  # 例如: "universal_poker(betting=nolimit,num_players=2,stack=200,blind=1)"
        self._env = rl_environment.Environment(self.game)

        self.player_id = player_id
        self.opponent_policies = opponent_policies
        self.ae_encoder = ae_encoder
        self.max_episode_length = max_episode_length

        # ---- OpenSpiel spec 轉成 Gym space ----
        info_state_size = self._env.observation_spec()["info_state"][self.player_id]  # int :contentReference[oaicite:0]{index=0}
        self.info_state_size = info_state_size

        num_actions = self._env.action_spec()["num_actions"]
        self.num_actions = num_actions

        # AE latent 作為 observation
        latent_dim = self.ae_encoder.latent_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(num_actions)

        # 狀態紀錄
        self.current_opponent_idx: Optional[int] = None
        self.current_time_step = None
        self.step_count = 0

    # ----------------- internal helpers -----------------

    def _encode_info_state(self, time_step) -> np.ndarray:
        """從 OpenSpiel time_step 拿 info_state -> AE latent。"""
        info_state = np.array(
            time_step.observations["info_state"][self.player_id],
            dtype=np.float32,
        )
        with torch.no_grad():
            x = torch.from_numpy(info_state).unsqueeze(0)  # [1, info_state_size]
            z = self.ae_encoder(x).squeeze(0).cpu().numpy().astype(np.float32)
        return z

    def _play_opponent_until_our_turn(self, time_step):
        """
        內部 loop：
        讓對手（Deep CFR）自動下到輪到我們，或是遊戲結束。
        """
        while (not time_step.last()) and (time_step.observations["current_player"] != self.player_id):
            opp = self.opponent_policies[self.current_opponent_idx]
            opp_action = opp(time_step)
            # rl_environment.Environment.step 只吃 [action]（目前 player 的 action）
            time_step = self._env.step([opp_action])
        return time_step

    # ----------------- Gym API -----------------

    def reset(self):
        # 每個 episode 抽一個 task = 對手
        self.current_opponent_idx = np.random.randint(len(self.opponent_policies))
        self.step_count = 0

        self._env = rl_environment.Environment(self.game)  # 重新建一個 env 比較乾淨
        time_step = self._env.reset()
        time_step = self._play_opponent_until_our_turn(time_step)

        self.current_time_step = time_step

        if time_step.last():
            # 極端情況：開局就 terminal（理論上不太會）
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._encode_info_state(time_step)

        return obs

    def step(self, action):
        assert self.current_time_step is not None, "Call reset() first."
        # 確保現在是我們的回合
        if self.current_time_step.observations["current_player"] != self.player_id:
            # 正常來說不應該發生
            self.current_time_step = self._play_opponent_until_our_turn(self.current_time_step)

        # 我方出牌
        time_step = self._env.step([int(action)])

        # 出完之後，可能直接終局，也可能輪到對手
        time_step = self._play_opponent_until_our_turn(time_step)

        self.current_time_step = time_step
        self.step_count += 1

        # reward 是 per-player 的 payoff
        reward = float(time_step.rewards[self.player_id])

        done = bool(time_step.last() or self.step_count >= self.max_episode_length)

        if done:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._encode_info_state(time_step)

        info = {
            "opponent_id": self.current_opponent_idx,
        }

        return obs, reward, done, info

# ============================================================
# 4. RL^2 Wrapper：把 (z_t, a_{t-1}, r_{t-1}, done_{t-1}) 變成 observation
# ============================================================

class RL2Wrapper(gym.Wrapper):
    """
    把 base env 的 observation = z_t
    包成 RL^2 style observation:
        [ z_t, one_hot(a_{t-1}), r_{t-1}, done_{t-1} ]
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Discrete)

        self.latent_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        # one-hot(action) + reward + done
        extra_dim = self.n_actions + 2

        low = -np.inf * np.ones(self.latent_dim + extra_dim, dtype=np.float32)
        high = np.inf * np.ones(self.latent_dim + extra_dim, dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # RL^2 history
        self.prev_action_one_hot = np.zeros(self.n_actions, dtype=np.float32)
        self.prev_reward = 0.0
        self.prev_done = 0.0

    def _build_obs(self, z: np.ndarray) -> np.ndarray:
        extra = np.concatenate(
            [
                self.prev_action_one_hot,
                np.array([self.prev_reward, self.prev_done], dtype=np.float32),
            ],
            axis=0,
        )
        return np.concatenate([z, extra], axis=0).astype(np.float32)

    def reset(self):
        # 新的一局 = 新的一個 RL^2 trajectory，history 歸零
        self.prev_action_one_hot[:] = 0.0
        self.prev_reward = 0.0
        self.prev_done = 0.0

        z = self.env.reset()
        return self._build_obs(z)

    def step(self, action):
        z_next, reward, done, info = self.env.step(action)

        # 更新 history（for 下一個時間步）
        self.prev_action_one_hot[:] = 0.0
        self.prev_action_one_hot[int(action)] = 1.0
        self.prev_reward = float(reward)
        self.prev_done = float(done)

        obs = self._build_obs(z_next)
        return obs, reward, done, info

# ============================================================
# 5. 建 Environment + RecurrentPPO Train Loop
# ============================================================

def make_poker_env(
    ae_ckpt_path: Optional[str],
    deep_cfr_policies: List,  # 你實際的 Deep CFR policy list
    latent_dim: int = 64,
    max_episode_length: int = 200,
) -> gym.Env:
    """
    回傳一個包好了 RL^2 的 Gym Env，給 SB3 用。
    """

    # ---- Step 1: 先建一個暫時的 env 只為了拿 info_state_size ----
    tmp_env = rl_environment.Environment(
        "universal_poker(betting=nolimit,num_players=2,stack=200,blind=1)"
    )
    info_state_size = tmp_env.observation_spec()["info_state"][0]
    del tmp_env

    # ---- Step 2: 載 AE ----
    ae_encoder = load_pretrained_ae(
        input_dim=info_state_size,
        latent_dim=latent_dim,
        ckpt_path=ae_ckpt_path,
    )

    # ---- Step 3: 包 Deep CFR 對手 ----
    opp_wrappers = [DeepCFRAgentWrapper(pi) for pi in deep_cfr_policies]

    base_env = OpenSpielPoker1v1Env(
        game_string="universal_poker(betting=nolimit,num_players=2,stack=200,blind=1)",
        opponent_policies=opp_wrappers,
        ae_encoder=ae_encoder,
        max_episode_length=max_episode_length,
        player_id=0,
    )

    rl2_env = RL2Wrapper(base_env)
    return rl2_env


def main():
    # TODO: 放你實際的 4 個 Deep CFR Agent instance
    # deep_cfr_policies = [deep_cfr_agent_0, ..., deep_cfr_agent_3]
    deep_cfr_policies = []  # placeholder

    if len(deep_cfr_policies) == 0:
        raise RuntimeError("請把 4 個 Deep CFR policy 接進來再跑。")

    def make_env_fn():
        return make_poker_env(
            ae_ckpt_path="ae_encoder.pt",  # TODO: 你的 AE checkpoint
            deep_cfr_policies=deep_cfr_policies,
            latent_dim=64,
            max_episode_length=200,
        )

    vec_env = DummyVecEnv([make_env_fn])

    # RecurrentPPO，預設用 LSTM
    model = RecurrentPPO(
        policy=MlpLstmPolicy,
        env=vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
    )

    model.learn(total_timesteps=1_000_000)
    model.save("rl2_poker_ppo_lstm")

    vec_env.close()


if __name__ == "__main__":
    main()
