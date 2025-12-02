import abc
import torch as tc

from rl2.agents.preprocessing.common import one_hot, Preprocessing

class VisionNet(abc.ABC, tc.nn.Module):
    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        pass

    @abc.abstractmethod
    def forward(self, curr_obs: tc.FloatTensor) -> tc.FloatTensor:
        pass

class MDPPreprocessing(Preprocessing):
    def __init__(self, num_actions: int, vision_net: VisionNet):
        super().__init__()
        self._num_actions = num_actions
        self._vision_net = vision_net

    @property
    def output_dim(self):
        return self._vision_net.output_dim + self._num_actions + 2

    def forward(
        self,
        curr_obs: tc.FloatTensor,
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor
    ) -> tc.FloatTensor:
        curr_obs_shape = list(curr_obs.shape)
        curr_obs = curr_obs.view(-1, *curr_obs_shape[-3:])
        emb_o = self._vision_net(curr_obs)
        emb_o = emb_o.view(*curr_obs_shape[:-3], emb_o.shape[-1])

        emb_a = one_hot(prev_action, depth=self._num_actions)
        prev_reward = prev_reward.unsqueeze(-1)
        prev_done = prev_done.unsqueeze(-1)
        vec = tc.cat(
            (emb_o, emb_a, prev_reward, prev_done), dim=-1).float()
        return vec
