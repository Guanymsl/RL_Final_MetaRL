import numpy as np

from agent.cfr.cfr import PPOAgent
from environment.param import MODE

aggresive = PPOAgent(model_path="agent/cfr/models/aggressive", deterministic=True)
passive   = PPOAgent(model_path="agent/cfr/models/passive", deterministic=True)
tight     = PPOAgent(model_path="agent/cfr/models/tight", deterministic=True)
loose     = PPOAgent(model_path="agent/cfr/models/loose", deterministic=True)
baseline  = PPOAgent(model_path="agent/cfr/models/baseline", deterministic=True)

class OpponentSampler:
    def __init__(self, mode=MODE):
        if (mode == 0):
            self.opponents = [
                aggresive,
                passive,
                tight,
                loose,
            ]
        elif (mode == 1):
            self.opponents = [
                baseline,
            ]

    def sample(self):
        return np.random.choice(self.opponents)
