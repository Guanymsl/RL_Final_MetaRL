import numpy as np

class AlwaysFold():
    def __init__(self):
        pass
    def step(self, state):
        legal_actions = list(state["legal_actions"].keys())
        return 2 if 2 in legal_actions else int(np.random.choice(legal_actions))

class AlwaysCall():
    def __init__(self):
        pass
    def step(self, state):
        legal_actions = list(state["legal_actions"].keys())
        return 0 if 0 in legal_actions else int(np.random.choice(legal_actions))
