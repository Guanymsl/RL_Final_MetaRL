import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class RL2FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super().__init__(observation_space, features_dim)
        self._features_dim = observation_space.shape[0]

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

        self.lstm_state = None

    def reset_lstm(self, n_envs=1):
        h = torch.zeros(1, n_envs, self.lstm_hidden_size).to(self.device)
        c = torch.zeros(1, n_envs, self.lstm_hidden_size).to(self.device)
        self.lstm_state = (h, c)

    def _prepare_lstm_input(self, obs):
        if obs.dim() == 2:
            return obs.unsqueeze(1)
        return obs

    def forward(self, obs, deterministic=False):
        obs = self._prepare_lstm_input(obs)

        lstm_out, self.lstm_state = self.lstm(obs, self.lstm_state)
        last = lstm_out[:, -1, :]

        logits = self.actor(last)
        values = self.critic(last)
        dist = self._get_action_dist_from_logits(logits)

        actions = dist.get_actions(deterministic=deterministic)
        log_probs = dist.log_prob(actions)

        return actions, values, log_probs

    def get_distribution(self, obs):
        obs = self._prepare_lstm_input(obs)
        lstm_out, _ = self.lstm(obs, self.lstm_state)
        last = lstm_out[:, -1, :]
        logits = self.actor(last)
        return self._get_action_dist_from_logits(logits)

    def forward_actor(self, obs):
        obs = self._prepare_lstm_input(obs)
        return self.get_distribution(obs).distribution

    def forward_critic(self, obs):
        obs = self._prepare_lstm_input(obs)
        lstm_out, _ = self.lstm(obs, self.lstm_state)
        last = lstm_out[:, -1, :]
        return self.critic(last)
