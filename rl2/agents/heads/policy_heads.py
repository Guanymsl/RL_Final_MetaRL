import torch as tc

class LinearPolicyHead(tc.nn.Module):
    def __init__(self, num_features, num_actions):
        super().__init__()
        self._num_features = num_features
        self._num_actions = num_actions
        self._linear = tc.nn.Linear(
            in_features=self._num_features,
            out_features=self._num_actions,
            bias=True)
        tc.nn.init.xavier_normal_(self._linear.weight)
        tc.nn.init.zeros_(self._linear.bias)

    def forward(self, features: tc.FloatTensor) -> tc.distributions.Categorical:
        logits = self._linear(features)
        dists = tc.distributions.Categorical(logits=logits)
        return dists
