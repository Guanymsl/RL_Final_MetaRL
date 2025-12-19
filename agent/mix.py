import random
import numpy as np

class MixtureOpponent:
    def __init__(self, agents, weights, noise_prob=0.1, temperature=1.5):
        self.agents = agents
        self.noise_prob = noise_prob

        w = np.array(weights, dtype=np.float32)
        w = w ** temperature
        self.weights = (w / w.sum()).tolist()

    def step(self, state):
        legal = list(state["legal_actions"].keys())

        if random.random() < self.noise_prob:
            return random.choice(legal)

        agent = random.choices(self.agents, weights=self.weights)[0]
        return agent.step(state)
