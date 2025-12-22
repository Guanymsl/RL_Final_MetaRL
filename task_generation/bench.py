#!/usr/bin/env python3
"""
Benchmark script for evaluating two agents against each other.

This script allows two agents (neural network .pt/.zip files or PKL .pkl files) 
to play head-to-head matches in RLCard Limit Hold'em and tracks statistics like 
win rates, average returns, etc.

Supports:
- Neural network agents (.pt files) - uses pt_wrapper.py
- Neural network agents (.zip files) - uses zip_wrapper.py
- PKL-based agents (.pkl files) - uses pkl_wrapper.py

Usage:
    # NN vs NN
    python bench.py --model1 cfr_bc_policy.pt --model2 another_model.pt --games 100
    
    # ZIP vs ZIP
    python bench.py --model1 model.zip --model2 another_model.zip --games 100
    
    # PKL vs PKL
    python bench.py --model1 cfr_model_avg_pol.pkl --model2 another_policy.pkl --games 100
    
    # Mixed: NN vs PKL
    python bench.py --model1 cfr_bc_policy.pt --model2 cfr_model_avg_pol.pkl --games 100
    
    # Mixed: ZIP vs PT
    python bench.py --model1 model.zip --model2 cfr_bc_policy.pt --games 100
"""

import argparse
import numpy as np
import rlcard
from typing import List, Dict, Tuple

from wrappers.pt_wrapper import NeuralNetworkAgent
from wrappers.zip_wrapper import PPOAgent
from wrappers.pkl_wrapper import PKLAgent


class ModelBenchmark:
    """Benchmark two agents (neural network or PKL-based) head-to-head."""

    def __init__(self, model_paths: List[str], device: str = "cpu"):
        """
        Initialize benchmark with multiple models.

        Supports .pt (neural network), .zip (neural network), and .pkl (CFR policy) files.

        Args:
            model_paths: List of paths to model files (.pt, .zip, or .pkl)
            device: Device to run inference on ("cpu" or "cuda") - only used for .pt and .zip files
        """
        self.models = []
        self.model_names = []
        self.model_types = []  # Track whether each model is "nn" or "pkl"

        for i, path in enumerate(model_paths):
            try:
                # Determine model type from file extension
                if path.endswith(".pt"):
                    agent = NeuralNetworkAgent(
                        model_path=path, device=device, deterministic=True
                    )
                    model_type = "nn"
                    name = (
                        path.split("/")[-1].replace(".pt", "").replace("_", " ").title()
                    )
                elif path.endswith(".zip"):
                    agent = PPOAgent(model_path=path, deterministic=True)
                    model_type = "nn"
                    name = (
                        path.split("/")[-1]
                        .replace(".zip", "")
                        .replace("_", " ")
                        .title()
                    )
                elif path.endswith(".pkl"):
                    agent = PKLAgent(pkl_path=path, deterministic=True)
                    model_type = "pkl"
                    name = (
                        path.split("/")[-1]
                        .replace(".pkl", "")
                        .replace("_", " ")
                        .title()
                    )
                else:
                    raise ValueError(
                        f"Unsupported file type: {path}. Must be .pt, .zip, or .pkl"
                    )

                self.models.append(agent)
                self.model_names.append(name)
                self.model_types.append(model_type)
                print(f"Loaded model {i+1} ({model_type.upper()}): {name} from {path}")
            except Exception as e:
                print(f"Error loading model {path}: {e}")
                raise

        if len(self.models) != 2:
            raise ValueError(
                "Must provide exactly 2 models for head-to-head benchmarking"
            )

        print(f"\nLoaded {len(self.models)} models for benchmarking")
        print(f"Model types: {dict(zip(self.model_names, self.model_types))}")

    def play_game(
        self, model1_idx: int, model2_idx: int, verbose: bool = False
    ) -> Tuple[float, float]:
        """
        Play a single game between two models.

        Args:
            model1_idx: Index of first model (plays as player 0)
            model2_idx: Index of second model (plays as player 1)
            verbose: If True, print game details

        Returns:
            Tuple of (model1_return, model2_return)
        """
        # Create environment (matching cfr.py setup)
        env = rlcard.make("limit-holdem", config={"allow_step_back": False})
        env.game.allowed_raise_num = 2

        # Get agents
        agent1 = self.models[model1_idx]
        agent2 = self.models[model2_idx]

        # Reset environment
        state, player_id = env.reset()

        step_count = 0
        max_steps = 1000  # Safety limit to prevent infinite loops

        while not env.is_over() and step_count < max_steps:
            # Debug: print progress for first game
            if step_count == 0:
                print(f"  Starting game (step 0)...", end="", flush=True)
            elif step_count % 50 == 0:
                print(f".", end="", flush=True)
            # Check if we have legal actions
            legal_actions = list(state["legal_actions"].keys())
            if not legal_actions:
                print(f"Warning: No legal actions at step {step_count}, ending game")
                break

            try:
                if player_id == 0:
                    # Model 1's turn
                    action = agent1.step(state)
                else:
                    # Model 2's turn
                    action = agent2.step(state)

                # Validate action is legal
                if action not in legal_actions:
                    print(
                        f"\nWarning: Invalid action {action}, legal actions: {legal_actions}. Using random legal action."
                    )
                    action = np.random.choice(legal_actions)

                if verbose:
                    action_names = {0: "Fold", 1: "Call", 2: "Raise", 3: "Check"}
                    print(
                        f"Step {step_count}: Player {player_id} ({self.model_names[model1_idx if player_id == 0 else model2_idx]}) "
                        f"plays {action_names.get(action, action)}"
                    )

                state, player_id = env.step(action)
                step_count += 1

            except Exception as e:
                print(f"\nError during game at step {step_count}: {e}")
                print(f"State: {state}")
                print(f"Legal actions: {legal_actions}")
                raise

        if step_count > 0:
            print(f" (completed in {step_count} steps)")

        if step_count >= max_steps:
            print(f"Warning: Game exceeded {max_steps} steps, forcing termination")
            # Force game to end
            payoffs = [0.0, 0.0]  # Tie if we can't complete
        else:
            # Get payoffs
            payoffs = env.get_payoffs()

        model1_return = payoffs[0]
        model2_return = payoffs[1]

        return model1_return, model2_return

    def head_to_head(
        self, model1_idx: int, model2_idx: int, num_games: int = 100
    ) -> Dict:
        """
        Play head-to-head matches between two models.

        Args:
            model1_idx: Index of first model
            model2_idx: Index of second model
            num_games: Number of games to play

        Returns:
            Dictionary with statistics
        """
        model1_name = self.model_names[model1_idx]
        model2_name = self.model_names[model2_idx]

        print(f"\n{'='*60}")
        print(f"Head-to-Head: {model1_name} vs {model2_name}")
        print(f"Playing {num_games} games...")
        print(f"{'='*60}")

        model1_wins = 0
        model2_wins = 0
        ties = 0
        model1_total_return = 0.0
        model2_total_return = 0.0

        # Play games, alternating who goes first
        for game_num in range(num_games):
            if game_num % 2 == 0:
                # Model 1 goes first (player 0)
                ret1, ret2 = self.play_game(model1_idx, model2_idx)
            else:
                # Model 2 goes first (player 0) - swap positions
                ret2, ret1 = self.play_game(model2_idx, model1_idx)

            model1_total_return += ret1
            model2_total_return += ret2

            if ret1 > ret2:
                model1_wins += 1
            elif ret2 > ret1:
                model2_wins += 1
            else:
                ties += 1

            # Print progress more frequently
            if (game_num + 1) % 10 == 0 or (game_num + 1) == 1:
                print(
                    f"  Game {game_num + 1}/{num_games}: "
                    f"{model1_name} {model1_wins}-{model2_wins} {model2_name} "
                    f"(ties: {ties})"
                )

        # Calculate statistics
        stats = {
            "model1_name": model1_name,
            "model2_name": model2_name,
            "model1_wins": model1_wins,
            "model2_wins": model2_wins,
            "ties": ties,
            "model1_win_rate": model1_wins / num_games,
            "model2_win_rate": model2_wins / num_games,
            "model1_avg_return": model1_total_return / num_games,
            "model2_avg_return": model2_total_return / num_games,
            "model1_total_return": model1_total_return,
            "model2_total_return": model2_total_return,
            "num_games": num_games,
        }

        return stats

    def print_head_to_head_results(self, stats: Dict):
        """Print formatted head-to-head results."""
        print(f"\n{'='*60}")
        print("HEAD-TO-HEAD RESULTS")
        print(f"{'='*60}")
        print(f"{stats['model1_name']} vs {stats['model2_name']}")
        print(f"Games played: {stats['num_games']}")
        print(f"\nWins:")
        print(
            f"  {stats['model1_name']}: {stats['model1_wins']} ({stats['model1_win_rate']*100:.1f}%)"
        )
        print(
            f"  {stats['model2_name']}: {stats['model2_wins']} ({stats['model2_win_rate']*100:.1f}%)"
        )
        print(f"  Ties: {stats['ties']}")
        print(f"\nAverage Returns:")
        print(f"  {stats['model1_name']}: {stats['model1_avg_return']:.4f}")
        print(f"  {stats['model2_name']}: {stats['model2_avg_return']:.4f}")
        print(f"\nTotal Returns:")
        print(f"  {stats['model1_name']}: {stats['model1_total_return']:.4f}")
        print(f"  {stats['model2_name']}: {stats['model2_total_return']:.4f}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark two agents (.pt, .zip, or .pkl files) head-to-head"
    )
    parser.add_argument(
        "--model1",
        type=str,
        required=True,
        help="Path to first model file (.pt, .zip, or .pkl)",
    )
    parser.add_argument(
        "--model2",
        type=str,
        required=True,
        help="Path to second model file (.pt, .zip, or .pkl)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games per head-to-head match",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed game information",
    )
    args = parser.parse_args()

    # Validate arguments
    if not args.model1 or not args.model2:
        parser.error("Must provide both --model1 and --model2")

    # Create benchmark
    model_paths = [args.model1, args.model2]
    benchmark = ModelBenchmark(model_paths, device=args.device)

    # Head-to-head
    stats = benchmark.head_to_head(0, 1, num_games=args.games)
    benchmark.print_head_to_head_results(stats)


if __name__ == "__main__":
    main()
