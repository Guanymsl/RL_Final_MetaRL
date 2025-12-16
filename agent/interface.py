import numpy as np

from agent.cfr.cfr import PPOAgent

cfr_1 = PPOAgent(model_path="agent/cfr/models/cfr_agent_1", deterministic=True)
cfr_2 = PPOAgent(model_path="agent/cfr/models/cfr_agent_1", deterministic=True)
cfr_3 = PPOAgent(model_path="agent/cfr/models/cfr_agent_1", deterministic=True)
cfr_4 = PPOAgent(model_path="agent/cfr/models/cfr_agent_1", deterministic=True)

class OpponentSampler:
    def __init__(self):
        self.opponents = [
            cfr_1,
            cfr_2,
            cfr_3,
            cfr_4,
        ]

    def sample(self):
        return np.random.choice(self.opponents)
