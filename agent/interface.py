import random
from rlcard.agents import RandomAgent

from agent.cfr.cfr import PPOAgent
from agent.other.other import AlwaysFold, AlwaysCall

aggressive = PPOAgent(model_path="agent/cfr/models/aggressive", deterministic=True)
passive    = PPOAgent(model_path="agent/cfr/models/passive", deterministic=True)
tight      = PPOAgent(model_path="agent/cfr/models/tight", deterministic=True)
loose      = PPOAgent(model_path="agent/cfr/models/loose", deterministic=True)
baseline   = PPOAgent(model_path="agent/cfr/models/baseline", deterministic=True)

rand       = RandomAgent(num_actions=4)
fold       = AlwaysFold()
call       = AlwaysCall()

AGENTS = {
    "aggressive": aggressive,
    "passive": passive,
    "tight": tight,
    "loose": loose,
    "baseline": baseline,
    "rand": rand,
    "fold": fold,
    "call": call,
}

def task_sample():
    r = random.random()

    if r < 0.15:
        return random.choice([fold, call, rand])

    if r < 0.90:
        return random.choice([aggressive, passive, tight, loose])

    return baseline

class OpponentSampler:
    def __init__(self, opponent, mode='train'):
        self.mode = mode

        if mode == 'inference':
            self.opponent = AGENTS[opponent]
        else:
            self.opponent = None

    def sample(self):
        if self.mode == 'train':
            return task_sample()
        else:
            return self.opponent
