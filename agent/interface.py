import numpy as np
import random
from rlcard.agents import RandomAgent

from agent.cfr.cfr import PPOAgent
from agent.other.other import AlwaysFold, AlwaysCall
from agent.mix import MixtureOpponent

aggressive = PPOAgent(model_path="agent/cfr/models/aggressive", deterministic=True)
passive    = PPOAgent(model_path="agent/cfr/models/passive", deterministic=True)
tight      = PPOAgent(model_path="agent/cfr/models/tight", deterministic=True)
loose      = PPOAgent(model_path="agent/cfr/models/loose", deterministic=True)
baseline   = PPOAgent(model_path="agent/cfr/models/baseline", deterministic=True)

rand       = RandomAgent(num_actions=4)
fold       = AlwaysFold()
call       = AlwaysCall()

BASE_AGENTS = [
    aggressive,
    passive,
    tight,
    loose,
    baseline,
]

def task_sample():
    r = random.random()

    if r < 0.10:
        return random.choice([fold, call, rand])

    if r < 0.40:
        a, b = random.sample(BASE_AGENTS, 2)

        w = np.random.dirichlet([6.0, 1.0])
        return MixtureOpponent(
            agents=[a, b],
            weights=w,
            noise_prob=0.05,
            temperature=2.0,
        )

    if r < 0.90:
        alpha = [2.5, 2.5, 2.5, 2.5, 0.8]
        w = np.random.dirichlet(alpha)

        return MixtureOpponent(
            agents=BASE_AGENTS,
            weights=w,
            noise_prob=0.10,
            temperature=1.5,
        )

    a = random.choice(BASE_AGENTS)
    w = [0.85, 0.15]
    return MixtureOpponent(
        agents=[a, baseline],
        weights=w,
        noise_prob=0.05,
        temperature=2.5,
    )

class OpponentSampler:
    def __init__(self, mode='train'):
        self.mode = mode

    def sample(self):
        if self.mode == 'train':
            return task_sample()
        else:
            return baseline
