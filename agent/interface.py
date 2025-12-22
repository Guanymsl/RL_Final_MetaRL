import random
from rlcard.agents import RandomAgent

from agent.cfr.cfr import PPOAgent
from agent.other.other import AlwaysFold, AlwaysCall
from agent.other.parametric import ParametricAgent
from agent.other.human import ManualAgent

aggressive = PPOAgent(model_path="agent/cfr/models/aggressive", deterministic=True)
passive    = PPOAgent(model_path="agent/cfr/models/passive", deterministic=True)
tight      = PPOAgent(model_path="agent/cfr/models/tight", deterministic=True)
loose      = PPOAgent(model_path="agent/cfr/models/loose", deterministic=True)
baseline   = PPOAgent(model_path="agent/cfr/models/baseline", deterministic=True)

rand       = RandomAgent(num_actions=4)
fold       = AlwaysFold()
call       = AlwaysCall()

human      = ManualAgent()

def agent_sample():
    return ParametricAgent(
        random.random(),
        random.random(),
        random.random(),
        random.random(),
        random.random(),
    )

AGENTS = {
    "aggressive": aggressive,
    "passive": passive,
    "tight": tight,
    "loose": loose,
    "baseline": baseline,
    "rand": rand,
    "fold": fold,
    "call": call,
    "human": human,
}

# discrete: r < 0.15 fold call rand | 0.15 < r < 0.9 aggressive passive tight loose | 0.9 < r baseline
# continuous agent_sample()
# mix: r < 0.15 agent_sample() | 0.15 < r < 0.9 aggressive passive tight loose | 0.9 < r baseline
def task_sample():
    r = random.random()

    if r < 0.15:
        return agent_sample()

    if r < 0.9:
        return random.choice([aggressive, passive, tight, loose])

    return baseline

class OpponentSampler:
    def __init__(self, opponent, mode='train'):
        self.mode = mode

        if mode == 'inference':
            if opponent in AGENTS:
                self.opponent = AGENTS[opponent]
            else:
                self.opponent = agent_sample()
        else:
            self.opponent = None

    def sample(self):
        if self.mode == 'train':
            return task_sample()
        else:
            return self.opponent
