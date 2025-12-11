import numpy as np

from agent.cfr import CFR

class OpponentSampler:
    def __init__(self):
        self.opponents = [
            CFR(),
            CFR(),
            CFR(),
            CFR(),
        ]

    def sample(self):
        return np.random.choice(self.opponents)
