import torch as tc

class LinearValueHead(tc.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self._num_features = num_features
        self._linear = tc.nn.Linear(
            in_features=self._num_features,
            out_features=1,
            bias=True)
        tc.nn.init.xavier_normal_(self._linear.weight)
        tc.nn.init.zeros_(self._linear.bias)

    def forward(self, features: tc.FloatTensor) -> tc.FloatTensor:
        v_preds = self._linear(features).squeeze(-1)
        return v_preds
