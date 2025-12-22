# #!/usr/bin/env python3
# """
# PKL-based agent wrapper for RLCard environment.
#
# This module provides an agent class that loads a CFR policy from a .pkl file
# created by cfr.py. The agent uses the saved policy dictionary to select actions.
#
# Usage:
#     agent = PKLAgent(pkl_path="cfr_model_avg_pol.pkl")
#     action = agent.step(state)
# """
#
# import pickle
# import numpy as np
# from typing import Dict
#
#
# class PKLAgent:
#     """
#     Agent that uses a saved CFR policy from a .pkl file.
#
#     This agent loads a policy dictionary from a .pkl file (created by cfr.py)
#     and uses it to make decisions. The policy dictionary maps observation bytes
#     to action probability distributions.
#
#     Usage:
#         agent = PKLAgent(pkl_path="cfr_model_avg_pol.pkl", deterministic=False)
#         action = agent.step(state)
#     """
#
#     def __init__(
#         self,
#         pkl_path: str,
#         deterministic: bool = False
#     ):
#         """
#         Initialize the PKL-based agent.
#
#         Args:
#             pkl_path: Path to the .pkl file containing the CFR policy dictionary
#             deterministic: If True, always select the highest probability action.
#                           If False, sample from the action distribution.
#         """
#         self.deterministic = deterministic
#
#         # Load the .pkl file
#         print(f"Loading PKL policy from {pkl_path}...")
#         with open(pkl_path, "rb") as f:
#             self.policy = pickle.load(f)
#
#         # Count unique observations
#         num_states = len(self.policy)
#         print(f"Loaded policy with {num_states} unique states")
#
#         # Check policy structure and normalize if needed
#         if num_states > 0:
#             first_key = next(iter(self.policy.keys()))
#             first_value = self.policy[first_key]
#             if isinstance(first_value, dict):
#                 print(f"Policy format: dict[bytes] -> dict[action_id: prob]")
#                 self._policy_is_dict = True
#             elif isinstance(first_value, np.ndarray):
#                 print(f"Policy format: dict[bytes] -> np.ndarray (probabilities)")
#                 self._policy_is_dict = False
#                 # Normalize arrays if needed (CFR policies might be unnormalized sums)
#                 self._normalize_policy_arrays()
#             else:
#                 print(f"Warning: Unexpected policy value type: {type(first_value)}")
#                 self._policy_is_dict = False
#         else:
#             self._policy_is_dict = False
#
#     def _normalize_policy_arrays(self):
#         """Normalize array-based policy probabilities if needed."""
#         # CFR policies might store unnormalized sums, so normalize them
#         for obs_bytes, probs_array in self.policy.items():
#             if isinstance(probs_array, np.ndarray):
#                 total = probs_array.sum()
#                 if total > 1e-8:
#                     # Normalize in-place
#                     self.policy[obs_bytes] = probs_array / total
#                 else:
#                     # Uniform if no probability mass
#                     self.policy[obs_bytes] = np.ones_like(probs_array) / len(probs_array)
#
#     def step(self, state: Dict) -> int:
#         """
#         Select an action given the current game state.
#
#         This method is compatible with RLCard's agent interface.
#
#         Args:
#             state: RLCard state dictionary containing:
#                 - "obs": observation array
#                 - "legal_actions": dictionary of legal actions
#
#         Returns:
#             int: Selected action
#         """
#         # Extract observation and legal actions from RLCard state
#         obs = state["obs"]
#         legal_actions = list(state["legal_actions"].keys())
#
#         if not legal_actions:
#             raise ValueError("No legal actions available")
#
#         # Convert observation to bytes for lookup
#         state_key = obs.tobytes()
#
#         if state_key in self.policy:
#             # Use CFR strategy
#             action_probs = self.policy[state_key]
#
#             # Handle both dict and array formats
#             if isinstance(action_probs, dict):
#                 # Filter illegal actions
#                 filtered_probs = {a: action_probs.get(a, 0.0) for a in legal_actions}
#                 # Normalize probabilities
#                 total = sum(filtered_probs.values())
#                 if total > 1e-8:
#                     if self.deterministic:
#                         # Select action with highest probability
#                         best_action = max(filtered_probs, key=filtered_probs.get)
#                         return best_action
#                     else:
#                         # Sample from the action distribution
#                         actions = list(filtered_probs.keys())
#                         probs = np.array([filtered_probs[a] / total for a in actions])
#                         return np.random.choice(actions, p=probs)
#             elif isinstance(action_probs, np.ndarray):
#                 # Extract probabilities for legal actions
#                 legal_probs = action_probs[legal_actions]
#                 # Normalize
#                 total = legal_probs.sum()
#                 if total > 1e-8:
#                     legal_probs = legal_probs / total
#                     if self.deterministic:
#                         # Select action with highest probability
#                         best_idx = np.argmax(legal_probs)
#                         return legal_actions[best_idx]
#                     else:
#                         # Sample from the action distribution
#                         return np.random.choice(legal_actions, p=legal_probs)
#
#         # Fallback: choose uniformly among legal actions
#         if self.deterministic:
#             # Return first legal action (arbitrary but deterministic)
#             return legal_actions[0]
#         else:
#             return np.random.choice(legal_actions)
#
#     def eval_step(self, state: Dict) -> int:
#         """
#         Evaluation step - same as step() but with deterministic=True.
#         This is the standard RLCard agent interface method.
#
#         Args:
#             state: RLCard state dictionary
#
#         Returns:
#             int: Selected action (deterministic)
#         """
#         # Temporarily set deterministic mode
#         original_deterministic = self.deterministic
#         self.deterministic = True
#         action = self.step(state)
#         self.deterministic = original_deterministic
#         return action
#
#
# # Example usage and testing
# if __name__ == "__main__":
#     import rlcard
#
#     # Create a test environment
#     env = rlcard.make("limit-holdem", config={"allow_step_back": False})
#     env.game.allowed_raise_num = 2
#
#     # Load agent
#     agent = PKLAgent(pkl_path="cfr_model_avg_pol.pkl", deterministic=False)
#
#     # Test the agent
#     state, player_id = env.reset()
#     print(f"Initial state: player_id={player_id}")
#     print(f"Legal actions: {list(state['legal_actions'].keys())}")
#
#     action = agent.step(state)
#     print(f"Selected action: {action}")
#
#     # Play a few steps
#     for i in range(5):
#         if env.is_over():
#             break
#         state, player_id = env.step(action)
#         if not env.is_over():
#             action = agent.step(state)
#             print(f"Step {i+1}: Player {player_id} plays action {action}")
#


#!/usr/bin/env python3
"""
PKL-based agent wrapper for RLCard environment.
Loads a CFR policy saved as a .pkl dictionary and performs action lookup.
"""

import pickle
import numpy as np
from typing import Dict


class PKLAgent:
    """
    Agent that uses a saved CFR policy from a .pkl file.

    The policy dictionary maps observation bytes -> action probability vectors.
    RLCard observations must be converted to float64 and re-serialized to match
    CFR’s state_key encoding.
    """

    def __init__(self, pkl_path: str, deterministic: bool = False):
        """
        Args:
            pkl_path: path to CFR policy .pkl file
            deterministic: if True, pick argmax action; else sample
        """
        self.deterministic = deterministic

        print(f"Loading PKL policy from {pkl_path}...")
        with open(pkl_path, "rb") as f:
            self.policy = pickle.load(f)

        print(f"Loaded policy with {len(self.policy)} unique states")

        # Track lookup hits/misses
        self.lookup_hits = 0
        self.lookup_misses = 0

        # Detect policy format
        if len(self.policy) > 0:
            first_value = next(iter(self.policy.values()))
            if isinstance(first_value, dict):
                print("Policy format: dict[bytes] -> dict[action_id: prob]")
                self._policy_is_dict = True
            elif isinstance(first_value, np.ndarray):
                print("Policy format: dict[bytes] -> np.ndarray(probabilities)")
                self._policy_is_dict = False
                self._normalize_policy_arrays()
            else:
                print(f"Warning: unexpected policy entry type: {type(first_value)}")
                self._policy_is_dict = False
        else:
            self._policy_is_dict = False

    def _normalize_policy_arrays(self):
        """Normalize array-based probs (CFR stores cumulative sums)."""
        print("Normalizing CFR probability arrays...")
        for key, arr in self.policy.items():
            total = arr.sum()
            if total > 1e-12:
                self.policy[key] = arr / total
            else:
                self.policy[key] = np.ones_like(arr) / len(arr)
        print("Done normalizing.\n")

    def step(self, state: Dict) -> int:
        """
        Choose an action using CFR policy lookup.

        Args:
            state: RLCard observation dictionary

        Returns:
            int: the chosen action ID
        """
        obs = state["obs"]
        legal_actions = list(state["legal_actions"].keys())

        if not legal_actions:
            raise ValueError("No legal actions available")

        # IMPORTANT FIX: Convert obs to float64 to match CFR encoding
        state_key = obs.astype(np.float64).tobytes()

        if state_key in self.policy:
            self.lookup_hits += 1
            entry = self.policy[state_key]

            if self._policy_is_dict:
                # Policy is stored as dict[action_id] = prob
                probs = np.array([entry.get(a, 0.0) for a in legal_actions])
            else:
                # Policy stored as a numpy array
                probs = entry[legal_actions]

            total = probs.sum()
            if total > 1e-12:
                probs = probs / total
            else:
                probs = np.ones(len(legal_actions)) / len(legal_actions)

            if self.deterministic:
                return legal_actions[np.argmax(probs)]
            else:
                return np.random.choice(legal_actions, p=probs)

        else:
            # State not found in CFR table
            self.lookup_misses += 1

            # Fallback: uniform among legal actions
            if self.deterministic:
                return legal_actions[0]
            return np.random.choice(legal_actions)

    def eval_step(self, state: Dict) -> int:
        """Deterministic evaluation action."""
        return self.step(state)


# ---------------- Test Example -----------------

if __name__ == "__main__":
    import rlcard

    env = rlcard.make("limit-holdem", config={"allow_step_back": False})
    env.game.allowed_raise_num = 2

    agent = PKLAgent("cfr_model_avg_pol.pkl", deterministic=False)

    state, player_id = env.reset()
    print(
        f"Initial player: {player_id}, legal actions: {list(state['legal_actions'].keys())}"
    )

    for t in range(5):
        if env.is_over():
            break

        if player_id == 0:
            action = agent.step(state)
        else:
            action = np.random.choice(list(state["legal_actions"].keys()))

        state, player_id = env.step(action)
        print(f"Step {t}, player {player_id}, action {action}")

    print("\nCFR lookup hits:", agent.lookup_hits)
    print("CFR lookup misses:", agent.lookup_misses)
    print("Payoffs:", env.get_payoffs())
