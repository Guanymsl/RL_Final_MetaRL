import numpy as np

from agent.cfr.cfr import PPOAgent

aggresive = PPOAgent(model_path="agent/cfr/models/aggressive", deterministic=True)
passive   = PPOAgent(model_path="agent/cfr/models/passive", deterministic=True)
tight     = PPOAgent(model_path="agent/cfr/models/tight", deterministic=True)
loose     = PPOAgent(model_path="agent/cfr/models/loose", deterministic=True)
baseline  = PPOAgent(model_path="agent/cfr/models/baseline", deterministic=True)

class OpponentSampler:
    def __init__(self, mode='train'):
        if mode == 'train':
            self.opponents = [
                aggresive,
                passive,
                tight,
                loose,
            ]

        else:
            self.opponents = [
                baseline,
            ]

    def sample(self):
        return np.random.choice(self.opponents)
