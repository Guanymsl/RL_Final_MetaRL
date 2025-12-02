import torch as tc
from typing import Union, Tuple, Optional, TypeVar, Generic

ArchitectureState = TypeVar('ArchitectureState')

class StatefulPolicyNet(tc.nn.Module, Generic[ArchitectureState]):
    def __init__(self, preprocessing, architecture, policy_head):
        super().__init__()
        self._preprocessing = preprocessing
        self._architecture = architecture
        self._policy_head = policy_head

    def initial_state(self, batch_size: int) -> Optional[ArchitectureState]:
        return self._architecture.initial_state(batch_size=batch_size)

    def forward(
        self,
        curr_obs: Union[tc.LongTensor, tc.FloatTensor],
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor,
        prev_state: Optional[ArchitectureState]
    ) -> Tuple[tc.distributions.Categorical, ArchitectureState]:
        inputs = self._preprocessing(
            curr_obs, prev_action, prev_reward, prev_done)

        features, new_state = self._architecture(
            inputs=inputs, prev_state=prev_state)

        dist = self._policy_head(features)

        return dist, new_state
