import abc
import torch as tc
from typing import Union

class Preprocessing(abc.ABC, tc.nn.Module):
    def forward(
        self,
        curr_obs: Union[tc.LongTensor, tc.FloatTensor],
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor
    ) -> tc.FloatTensor:
        pass

def one_hot(ys: tc.LongTensor, depth: int) -> tc.FloatTensor:
    vecs_shape = list(ys.shape) + [depth]
    vecs = tc.zeros(dtype=tc.float32, size=vecs_shape)
    vecs.scatter_(dim=-1, index=ys.unsqueeze(-1),
                  src=tc.ones(dtype=tc.float32, size=vecs_shape))
    return vecs.float()
