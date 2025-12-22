import numpy as np
import torch
import torch.nn as nn
import rlcard


class PokerPolicyNet(nn.Module):
    """
    Neural network architecture matching train.py.
    This must match the architecture used during training.
    """

    def __init__(self, obs_dim, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class NeuralNetworkAgent:
    """
    Wrapper class for a neural network that allows inference in the RLCard environment.

    This class loads a trained neural network model and provides an interface
    compatible with RLCard's agent API for playing Limit Hold'em.

    Usage:
        agent = NeuralNetworkAgent(model_path="cfr_bc_policy.pt")
        action = agent.step(state)
    """

    def __init__(
        self, model_path="cfr_bc_policy.pt", device="cpu", deterministic=False
    ):
        """
        Initialize the neural network agent.

        Args:
            model_path: Path to the saved model checkpoint (.pt file)
            device: Device to run inference on ("cpu" or "cuda")
            deterministic: If True, always select the highest probability action.
                          If False, sample from the action distribution.
        """
        self.device = torch.device(device)
        self.deterministic = deterministic
        self._dim_warning_shown = (
            False  # Track if we've shown the dimension mismatch warning
        )

        # Load checkpoint first to infer dimensions from the saved model
        checkpoint = torch.load(model_path, map_location=self.device)

        # Infer dimensions from checkpoint weights
        # The first layer weight shape is [hidden_dim, input_dim]
        first_layer_weight = checkpoint["net.0.weight"]
        self.obs_dim = first_layer_weight.shape[1]

        # The last layer weight shape is [output_dim, hidden_dim]
        last_layer_weight = checkpoint["net.6.weight"]
        self.action_dim = last_layer_weight.shape[0]

        print(
            f"Inferred from checkpoint: obs_dim={self.obs_dim}, action_dim={self.action_dim}"
        )

        # Initialize and load the neural network with correct dimensions
        self.model = PokerPolicyNet(self.obs_dim, self.action_dim)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        print(f"Loaded neural network model from {model_path}")
        print(
            f"Observation dimension: {self.obs_dim}, Action dimension: {self.action_dim}"
        )

    def step(self, state):
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
        # The model was trained with obs_dim=144, but current env might give different size
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
            # Only show warning once to avoid spam
            if not self._dim_warning_shown:
                print(
                    f"Note: Observation dimension mismatch. Model expects {self.obs_dim}, "
                    f"environment provides {obs_len}. Using {'padded' if obs_len < self.obs_dim else 'truncated'} observation. "
                    f"(This warning will not be shown again.)"
                )
                self._dim_warning_shown = True

        # Convert observation to tensor
        obs_tensor = torch.tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Get logits from neural network
        with torch.no_grad():
            logits = self.model(obs_tensor)
            logits = logits.squeeze(0)  # Remove batch dimension

        # Convert to probabilities using softmax
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Mask illegal actions by setting their probabilities to 0
        masked_probs = np.zeros(self.action_dim)
        for action in legal_actions:
            masked_probs[action] = probs[action]

        # Renormalize probabilities over legal actions
        prob_sum = masked_probs.sum()
        if prob_sum > 1e-8:
            masked_probs = masked_probs / prob_sum
        else:
            # Fallback: uniform distribution over legal actions
            masked_probs = np.zeros(self.action_dim)
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

        return action

    def eval_step(self, state):
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
    env = rlcard.make("limit-holdem", config={"allow_step_back": True})
    env.game.allowed_raise_num = 2
    print("Obs shape:", env.state_shape)

    # Create neural network agent
    agent = NeuralNetworkAgent(model_path="cfr_bc_policy.pt", deterministic=False)

    # Test the agent in the environment
    print("\nTesting neural network agent in RLCard environment...")
    state, player_id = env.reset()

    step_count = 0
    while not env.is_over():
        if player_id == 0:  # Neural network agent's turn
            action = agent.step(state)
            print(
                f"Step {step_count}: Player {player_id} (NN Agent) selected action {action}"
            )
        else:  # Random opponent
            action = np.random.choice(list(state["legal_actions"].keys()))
            print(
                f"Step {step_count}: Player {player_id} (Random) selected action {action}"
            )

        state, player_id = env.step(action)
        step_count += 1

    payoffs = env.get_payoffs()
    print(f"\nGame finished. Payoffs: {payoffs}")
    print(f"Neural network agent (player 0) payoff: {payoffs[0]}")
