import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class RL2FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super().__init__(observation_space, features_dim)

        self.obs_dim = observation_space.shape[0]
        self._features_dim = self.obs_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations

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

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.features_dim,
            hidden_size=lstm_hidden_size,
            batch_first=True,
        )

        self.actor = nn.Linear(lstm_hidden_size, self.action_space.n)
        self.critic = nn.Linear(lstm_hidden_size, 1)

        self.lstm_state = None

        self._initialize_weights()

    def reset_lstm(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.lstm_hidden_size).to(self.device)
        c = torch.zeros(1, batch_size, self.lstm_hidden_size).to(self.device)
        self.lstm_state = (h, c)

    def extract_features(self, obs):
        return obs

    def forward(self, obs, deterministic=False):
        lstm_out, self.lstm_state = self.lstm(obs, self.lstm_state)

        last_out = lstm_out[:, -1, :]

        logits = self.actor(last_out)
        values = self.critic(last_out)

        distribution = self._get_action_dist_from_logits(logits)

        if deterministic:
            actions = distribution.get_actions(deterministic=True)
        else:
            actions = distribution.get_actions()

        log_probs = distribution.log_prob(actions)

        return actions, values, log_probs

    def get_distribution(self, obs):
        lstm_out, _ = self.lstm(obs, self.lstm_state)
        last = lstm_out[:, -1, :]
        logits = self.actor(last)
        return self._get_action_dist_from_logits(logits)

    def forward_actor(self, obs):
        dist = self.get_distribution(obs)
        return dist.distribution

    def forward_critic(self, obs):
        lstm_out, _ = self.lstm(obs, self.lstm_state)
        last = lstm_out[:, -1, :]
        return self.critic(last)
