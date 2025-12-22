# from stable_baselines3 import PPO
# from env.env import HoldemTwoPlayerEnv
# import torch
#
# env = HoldemTwoPlayerEnv("../nn_models/base_policy.pt.pt")
#
# policy_kwargs = dict(
#     net_arch=dict(
#         pi=[512, 512, 256],  # policy network
#         vf=[512, 512, 256],  # value network (can be same or smaller)
#     ),
# )
#
#
# model = PPO(
#     "MlpPolicy",
#     env,
#     learning_rate=3e-5,  # small LR for fine-tuning
#     n_steps=2048,
#     batch_size=512,
#     n_epochs=10,
#     gamma=1.0,  # episodic poker
#     ent_coef=0.01,  # keeps exploration
#     clip_range=0.2,
#     policy_kwargs=policy_kwargs,
#     verbose=1,
# )
#
#
# base_policy = torch.load("cfr_bc_policy.pt", map_location="cpu")
#
# ppo_policy_net = model.policy.mlp_extractor.policy_net
#
# missing, unexpected = ppo_policy_net.load_state_dict(base_policy, strict=False)
#
# print("Missing keys:", missing)
# print("Unexpected keys:", unexpected)

#!/usr/bin/env python3

import os
import sys
import torch
from stable_baselines3 import PPO

# --------------------------------------------------
# Import environment
# --------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.env import HoldemTwoPlayerEnv


# --------------------------------------------------
# Configuration
# --------------------------------------------------
BC_MODEL_PATH = "nn_models/base/cfr_bc_policy.pt"
CHECKPOINT_DIR = "nn_models/base/base.zip"
TOTAL_TIMESTEPS = 1000000
WARMUP_STEPS = 20000
CHECKPOINT_EVERY = 100000

AGGRESSIVE = 0.0
TIGHT = 0.0

LEARNING_RATE = 1e-4


# --------------------------------------------------
# Create directories
# --------------------------------------------------
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# --------------------------------------------------
# Create environment
# --------------------------------------------------
env = HoldemTwoPlayerEnv(
    base_model_path=BC_MODEL_PATH,
    aggressive=AGGRESSIVE,
    tight=TIGHT,
)

print("Environment ready")
print("Observation dim:", env.observation_space.shape)
print("Action dim:", env.action_space.n)


# --------------------------------------------------
# PPO policy configuration (MUST match BC network)
# --------------------------------------------------
policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=dict(
        pi=[512, 512, 256],
        vf=[512, 512, 256],
    ),
)


# --------------------------------------------------
# Create PPO model
# --------------------------------------------------
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=LEARNING_RATE,
    n_steps=4096,
    batch_size=512,
    n_epochs=10,
    gamma=1.0,  # episodic poker
    ent_coef=0.01,
    clip_range=0.2,
    policy_kwargs=policy_kwargs,
    verbose=1,
)


# --------------------------------------------------
# Load BC weights into PPO policy
# --------------------------------------------------
print("\nLoading BC weights into PPO policy...")

bc_sd = torch.load(BC_MODEL_PATH, map_location="cpu")
bc_sd = {k.replace("net.", ""): v for k, v in bc_sd.items()}

# Load hidden layers
policy_net_sd = {k: v for k, v in bc_sd.items() if k.startswith(("0.", "2.", "4."))}
missing, unexpected = model.policy.mlp_extractor.policy_net.load_state_dict(
    policy_net_sd, strict=True
)
print("policy_net missing:", missing, "unexpected:", unexpected)

# Load action head
action_net_sd = {
    "weight": bc_sd["6.weight"],
    "bias": bc_sd["6.bias"],
}
missing, unexpected = model.policy.action_net.load_state_dict(
    action_net_sd, strict=True
)
print("action_net missing:", missing, "unexpected:", unexpected)

print("BC weights loaded successfully")


# --------------------------------------------------
# Warm-up: freeze policy, train value net only
# --------------------------------------------------
print("\nStarting warm-up phase (policy frozen)...")

for p in model.policy.mlp_extractor.policy_net.parameters():
    p.requires_grad = False
for p in model.policy.action_net.parameters():
    p.requires_grad = False

model.learn(total_timesteps=WARMUP_STEPS)

print("Warm-up finished, unfreezing policy")

for p in model.policy.mlp_extractor.policy_net.parameters():
    p.requires_grad = True
for p in model.policy.action_net.parameters():
    p.requires_grad = True


# --------------------------------------------------
# Main PPO fine-tuning loop with checkpoints
# --------------------------------------------------
trained_steps = WARMUP_STEPS

while trained_steps < TOTAL_TIMESTEPS:
    next_chunk = min(CHECKPOINT_EVERY, TOTAL_TIMESTEPS - trained_steps)

    print(
        f"\n=== PPO fine-tuning: steps {trained_steps} → {trained_steps + next_chunk} ==="
    )

    model.learn(
        total_timesteps=next_chunk,
        reset_num_timesteps=False,
    )

    trained_steps += next_chunk

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"ppo_step_{trained_steps}.zip")
    model.save(checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


# --------------------------------------------------
# Save final model
# --------------------------------------------------
final_path = os.path.join(CHECKPOINT_DIR, "ppo_final.zip")
model.save(final_path)

print("\n==============================")
print("PPO fine-tuning complete")
print(f"Final model saved to: {final_path}")
print("==============================")
