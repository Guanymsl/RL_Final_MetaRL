from stable_baselines3 import PPO
import numpy as np
import torch

class PPOAgent:
    def __init__(self, model_path, deterministic=False, device="cpu"):
        self.model = PPO.load(model_path, device=device)
        self.deterministic = deterministic
        self.device = device

    def step(self, state):
        obs = state["obs"].astype(np.float32)
        legal_actions = list(state["legal_actions"].keys())

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            features = self.model.policy.extract_features(obs_t)
            latent_pi, _ = self.model.policy.mlp_extractor(features)
            logits = self.model.policy.action_net(latent_pi).squeeze(0)

        logits = logits.cpu().numpy()

        masked_logits = np.full_like(logits, -1e9)
        masked_logits[legal_actions] = logits[legal_actions]

        probs = np.exp(masked_logits - masked_logits.max())
        probs /= probs.sum()

        if self.deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(len(probs), p=probs))

        return action

    def eval_step(self, state):
        return self.step(state)
