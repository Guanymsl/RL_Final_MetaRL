import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class RL2FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super().__init__(observation_space, features_dim)
        self._features_dim = features_dim

    def forward(self, obs):
        return obs

class RL2LstmPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        lstm_hidden_size=128,
        **kwargs
    ):
        self.lstm_hidden_size = lstm_hidden_size

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=RL2FeatureExtractor,
            features_extractor_kwargs=dict(features_dim=observation_space.shape[0]),
            **kwargs,
        )

        self.lstm = nn.LSTM(
            input_size=self.features_dim,
            hidden_size=lstm_hidden_size,
            batch_first=True,
        )

        self.actor = nn.Linear(lstm_hidden_size, self.action_space.n)
        self.critic = nn.Linear(lstm_hidden_size, 1)

        self.lstm_states = None
        self.init_lstm()

    def init_lstm(self):
        h = torch.zeros(1, 1, self.lstm_hidden_size, device=self.device)
        c = torch.zeros(1, 1, self.lstm_hidden_size, device=self.device)
        self.lstm_states = (h, c)

    def ensure_lstm_state(self, batch):
        h, c = self.lstm_states
        if h.size(1) != batch:
            self.lstm_states = (
                torch.zeros(1, batch, self.lstm_hidden_size, device=self.device),
                torch.zeros(1, batch, self.lstm_hidden_size, device=self.device)
            )

    def reset_lstm(self, env_idx):
        h, c = self.lstm_states
        h[:, env_idx, :] = 0.0
        c[:, env_idx, :] = 0.0

    def _prepare_lstm_input(self, obs):
        if obs.dim() == 2:
            return obs.unsqueeze(1)
        return obs

    def forward(self, obs, deterministic=False):
        obs = self._prepare_lstm_input(obs)
        self.ensure_lstm_state(obs.size(0))

        lstm_out, self.lstm_states = self.lstm(obs, self.lstm_states)
        last = lstm_out[:, -1, :]

        logits = self.actor(last)
        values = self.critic(last)
        dist = self._get_action_dist_from_logits(logits)

        actions = dist.get_actions(deterministic=deterministic)
        log_probs = dist.log_prob(actions)

        return actions, values, log_probs

    def get_distribution(self, obs):
        obs = self._prepare_lstm_input(obs)
        self.ensure_lstm_state(obs.size(0))

        lstm_out, _ = self.lstm(obs, self.lstm_states)
        last = lstm_out[:, -1, :]
        logits = self.actor(last)
        return self._get_action_dist_from_logits(logits)

    def forward_critic(self, obs):
        obs = self._prepare_lstm_input(obs)
        self.ensure_lstm_state(obs.size(0))

        lstm_out, _ = self.lstm(obs, self.lstm_states)
        last = lstm_out[:, -1, :]
        return self.critic(last)
