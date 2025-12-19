import numpy as np
from rlcard.agents import RandomAgent

from agent.cfr.cfr import PPOAgent
from agent.other.other import AlwaysFold, AlwaysCall

aggressive = PPOAgent(model_path="agent/cfr/models/aggressive", deterministic=True)
passive    = PPOAgent(model_path="agent/cfr/models/passive", deterministic=True)
tight      = PPOAgent(model_path="agent/cfr/models/tight", deterministic=True)
loose      = PPOAgent(model_path="agent/cfr/models/loose", deterministic=True)
baseline   = PPOAgent(model_path="agent/cfr/models/baseline", deterministic=True)
random     = RandomAgent(num_actions=4)
fold       = AlwaysFold()
call       = AlwaysCall()

class OpponentSampler:
    def __init__(self, mode='train'):
        if mode == 'train':
            self.opponents = [
                aggressive,
                passive,
                tight,
                loose,
                random,
                fold,
                call,
            ]

        else:
            self.opponents = [
                baseline,
            ]

    def sample(self):
        return np.random.choice(self.opponents)
