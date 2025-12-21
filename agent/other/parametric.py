import numpy as np
import random

class ParametricAgent:
    def __init__(
        self,
        p_call=0.25,
        p_raise=0.25,
        p_fold=0.25,
        p_check=0.25,
        temperature=1.0,
    ):
        self.base_pref = {
            0: p_call,
            1: p_raise,
            2: p_fold,
            3: p_check,
        }
        self.temperature = temperature
        self.rng = np.random.default_rng()

    def step(self, state):
        legal_actions = list(state["legal_actions"].keys())
        prefs = np.array([
            self.base_pref.get(a, 0.0)
            for a in legal_actions
        ], dtype=np.float32)

        if np.all(prefs == 0):
            return int(self.rng.choice(legal_actions))

        prefs = prefs / self.temperature
        exp_prefs = np.exp(prefs - np.max(prefs))
        probs = exp_prefs / exp_prefs.sum()

        return int(self.rng.choice(legal_actions, p=probs))
