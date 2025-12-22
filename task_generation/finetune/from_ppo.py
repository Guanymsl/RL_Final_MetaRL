import os
import sys
import torch
from stable_baselines3 import PPO

# --------------------------------------------------
# Import environment
# --------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.env import HoldemTwoPlayerEnv

BC_MODEL_PATH = "nn_models/base/base.zip"
TOTAL_TIMESTEPS = 1000000
WARMUP_STEPS = 20000
CHECKPOINT_EVERY = 100000

AGGRESSIVE = -10.0
TIGHT = 0.0
CHECKPOINT_DIR = f"nn_models/checkpoints/a{AGGRESSIVE}t{TIGHT}"

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

model = PPO.load(BC_MODEL_PATH, env=env)

trained_steps = 0
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

final_path = os.path.join(CHECKPOINT_DIR, "ppo_final.zip")
model.save(final_path)
print("\n==============================")
print("PPO fine-tuning complete")
print(f"Final model saved to: {final_path}")
print("==============================")
