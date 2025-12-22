#!/usr/bin/env python3
"""
NPZ-based agent wrapper for RLCard environment.

This module provides an agent class that loads action probabilities from a .npz file
created by data.py from CFR policy data. The agent uses a lookup table approach
to find matching observations and return stored action probabilities.

Usage:
    agent = NPZAgent(npz_path="cfr_bc_dataset.npz")
    action = agent.step(state)
"""

import numpy as np
import rlcard
from typing import Dict, List, Optional
from collections import defaultdict


class NPZAgent:
    """
    Agent that uses stored action probabilities from a .npz file.
    
    This agent loads observations and action probabilities from a .npz file
    (created by data.py) and uses them to make decisions. It finds the closest
    matching observation in the dataset and uses the associated action probabilities.
    
    Usage:
        agent = NPZAgent(npz_path="cfr_bc_dataset.npz")
        action = agent.step(state)
    """
    
    def __init__(
        self, 
        npz_path: str, 
        deterministic: bool = False,
        exact_match: bool = False
    ):
        """
        Initialize the NPZ-based agent.
        
        Args:
            npz_path: Path to the .npz file containing observations and probabilities
            deterministic: If True, always select the highest probability action.
                          If False, sample from the action distribution.
            exact_match: If True, only use exact observation matches.
                        If False, use nearest neighbor for approximate matches.
        """
        self.deterministic = deterministic
        self.exact_match = exact_match
        
        # Load the .npz file
        print(f"Loading NPZ dataset from {npz_path}...")
        data = np.load(npz_path)
        self.obs_array = data["obs"]  # Shape: (N, obs_dim)
        self.probs_array = data["probs"]  # Shape: (N, action_dim)
        
        self.num_samples = len(self.obs_array)
        self.obs_dim = self.obs_array.shape[1]
        self.action_dim = self.probs_array.shape[1]
        
        print(f"Loaded {self.num_samples} samples")
        print(f"Observation dimension: {self.obs_dim}, Action dimension: {self.action_dim}")
        
        # Create lookup dictionary for exact matches (faster)
        # Use observation as bytes for hashing
        self.obs_lookup: Dict[bytes, np.ndarray] = {}
        print("Building lookup table...", end="", flush=True)
        for i in range(self.num_samples):
            obs_bytes = self.obs_array[i].tobytes()
            self.obs_lookup[obs_bytes] = self.probs_array[i]
            if (i + 1) % 10000 == 0:
                print(f".", end="", flush=True)
        print(f" Done!")
        print(f"Created lookup table with {len(self.obs_lookup)} unique observations")
        
        # Cache for nearest neighbor lookups to speed up repeated queries
        self._nn_cache: Dict[bytes, int] = {}
        
        # Pre-sample indices for large datasets to speed up nearest neighbor search
        if self.num_samples > 50000:
            sample_size = min(10000, self.num_samples)
            # Use a fixed random seed for reproducibility
            np.random.seed(42)
            self._nn_sample_indices = np.random.choice(
                self.num_samples, size=sample_size, replace=False
            )
            self._nn_sample_array = self.obs_array[self._nn_sample_indices]
            print(f"Using approximate nearest neighbor with {sample_size} samples")
        else:
            self._nn_sample_indices = None
            self._nn_sample_array = None
    
    def _find_closest_observation(self, obs: np.ndarray) -> int:
        """
        Find the index of the closest observation in the dataset.
        
        Uses caching to speed up repeated queries for the same observation.
        For large datasets, uses a faster approximate method.
        
        Args:
            obs: Observation array to match
        
        Returns:
            Index of closest observation in dataset
        """
        # Check cache first
        obs_bytes = obs.tobytes()
        if obs_bytes in self._nn_cache:
            return self._nn_cache[obs_bytes]
        
        # For very large datasets, use pre-sampled subset for faster search
        if self._nn_sample_array is not None:
            # Use pre-sampled subset for approximate nearest neighbor
            obs_expanded = obs[np.newaxis, :]  # Shape: (1, obs_dim)
            distances = np.linalg.norm(self._nn_sample_array - obs_expanded, axis=1)
            closest_sample_idx = int(np.argmin(distances))
            closest_idx = int(self._nn_sample_indices[closest_sample_idx])
        else:
            # For smaller datasets, search all observations
            # Compute L2 distances to all observations
            # obs_array shape: (N, obs_dim), obs shape: (obs_dim,)
            # Use vectorized computation for efficiency
            obs_expanded = obs[np.newaxis, :]  # Shape: (1, obs_dim)
            distances = np.linalg.norm(self.obs_array - obs_expanded, axis=1)
            closest_idx = int(np.argmin(distances))
        
        # Cache the result
        self._nn_cache[obs_bytes] = closest_idx
        
        return closest_idx
    
    def step(self, state: Dict) -> int:
        """
        Select an action given the current game state.
        
        This method is compatible with RLCard's agent interface.
        
        Args:
            state: RLCard state dictionary containing:
                - "obs": observation array
                - "legal_actions": dictionary of legal actions
        
        Returns:
            int: Selected action
        """
        # Extract observation and legal actions from RLCard state
        obs = state["obs"]
        legal_actions = list(state["legal_actions"].keys())
        
        if not legal_actions:
            raise ValueError("No legal actions available")
        
        # Handle observation dimension mismatch
        obs_len = len(obs)
        if obs_len != self.obs_dim:
            if obs_len < self.obs_dim:
                # Pad with zeros if observation is smaller
                obs = np.pad(
                    obs, (0, self.obs_dim - obs_len), mode="constant", constant_values=0
                )
            else:
                # Truncate if observation is larger
                obs = obs[: self.obs_dim]
        
        # Try exact match first
        obs_bytes = obs.tobytes()
        if obs_bytes in self.obs_lookup:
            probs = self.obs_lookup[obs_bytes].copy()
        elif self.exact_match:
            # If exact match required but not found, use uniform over legal actions
            probs = np.zeros(self.action_dim, dtype=np.float32)
            for action in legal_actions:
                probs[action] = 1.0 / len(legal_actions)
        else:
            # Find closest observation using nearest neighbor
            # This can be slow for large datasets, so we optimize it
            try:
                closest_idx = self._find_closest_observation(obs)
                probs = self.probs_array[closest_idx].copy()
            except Exception as e:
                # Fallback to uniform if nearest neighbor fails
                print(f"Warning: Nearest neighbor search failed: {e}, using uniform policy")
                probs = np.zeros(self.action_dim, dtype=np.float32)
                for action in legal_actions:
                    probs[action] = 1.0 / len(legal_actions)
        
        # Mask illegal actions by setting their probabilities to 0
        masked_probs = np.zeros(self.action_dim, dtype=np.float32)
        for action in legal_actions:
            masked_probs[action] = probs[action]
        
        # Renormalize probabilities over legal actions
        prob_sum = masked_probs.sum()
        if prob_sum > 1e-8:
            masked_probs = masked_probs / prob_sum
        else:
            # Fallback: uniform distribution over legal actions
            masked_probs = np.zeros(self.action_dim, dtype=np.float32)
            for action in legal_actions:
                masked_probs[action] = 1.0 / len(legal_actions)
        
        # Select action
        if self.deterministic:
            # Select action with highest probability among legal actions
            legal_probs = np.array([masked_probs[a] for a in legal_actions])
            best_idx = np.argmax(legal_probs)
            action = legal_actions[best_idx]
        else:
            # Sample from the action distribution
            action = np.random.choice(self.action_dim, p=masked_probs)
        
        return int(action)
    
    def eval_step(self, state: Dict) -> int:
        """
        Evaluation step - same as step() but with deterministic=True.
        This is the standard RLCard agent interface method.
        """
        # Temporarily set deterministic mode
        original_deterministic = self.deterministic
        self.deterministic = True
        action = self.step(state)
        self.deterministic = original_deterministic
        return action


# Example usage and testing
if __name__ == "__main__":
    # Create RLCard environment (matching cfr.py setup)
    env = rlcard.make("limit-holdem", config={"allow_step_back": False})
    env.game.allowed_raise_num = 2
    
    # Create NPZ agent
    agent = NPZAgent(npz_path="../cfr_bc_dataset.npz", deterministic=False)
    
    # Test the agent in the environment
    print("\nTesting NPZ agent in RLCard environment...")
    state, player_id = env.reset()
    
    step_count = 0
    while not env.is_over():
        if player_id == 0:  # NPZ agent's turn
            action = agent.step(state)
            print(f"Step {step_count}: Player {player_id} (NPZ Agent) selected action {action}")
        else:  # Random opponent
            action = np.random.choice(list(state["legal_actions"].keys()))
            print(f"Step {step_count}: Player {player_id} (Random) selected action {action}")
        
        state, player_id = env.step(action)
        step_count += 1
    
    payoffs = env.get_payoffs()
    print(f"\nGame finished. Payoffs: {payoffs}")
    print(f"NPZ agent (player 0) payoff: {payoffs[0]}")

