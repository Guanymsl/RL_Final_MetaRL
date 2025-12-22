import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

TRAIN_EPOCHS = 30


# ----------------------
# Dataset Loader
# ----------------------
class CFRDataset(Dataset):
    def __init__(self, path="../dataset/cfr_bc_dataset.npz"):
        data = np.load(path)
        self.obs = data["obs"]  # shape (N, obs_dim)
        self.probs = data["probs"]  # shape (N, action_dim)

        # Log dataset stats
        print("=== Dataset Loaded ===")
        print("obs dtype:", self.obs.dtype)
        print("probs dtype:", self.probs.dtype)
        print("obs shape:", self.obs.shape)
        print("probs shape:", self.probs.shape)
        print("obs_dim =", self.obs.shape[1])
        print("action_dim =", self.probs.shape[1])
        print("======================\n")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.obs[idx], dtype=torch.float32),
            torch.tensor(self.probs[idx], dtype=torch.float32),
        )


# ----------------------
# Policy Network
# ----------------------
class PokerPolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim=4):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        print("=== NN Architecture ===")
        print(f"Input dimension:  {obs_dim}")
        print(f"Output dimension: {action_dim}")
        print("=======================\n")

    def forward(self, x):
        # Runtime dimension check
        if x.shape[1] != self.obs_dim:
            raise ValueError(
                f"NN forward(): Expected obs_dim {self.obs_dim}, but got {x.shape[1]}"
            )
        return self.net(x)


# ----------------------
# Load dataset
# ----------------------
dataset = CFRDataset("../dataset/cfr_bc_dataset.npz")
loader = DataLoader(dataset, batch_size=4096, shuffle=True, drop_last=True)

obs_dim = dataset.obs.shape[1]
action_dim = dataset.probs.shape[1]

# ----------------------
# Create model + optimizer
# ----------------------
model = PokerPolicyNet(obs_dim, action_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


# ----------------------
# Soft cross-entropy loss
# ----------------------
def soft_cross_entropy(logits, target_probs):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


# ----------------------
# Training Loop
# ----------------------
for epoch in range(TRAIN_EPOCHS):
    total_loss = 0.0

    for batch_index, (obs, probs) in enumerate(loader):
        # Logging: check batch dimensions
        if batch_index == 0:
            print(f"Epoch {epoch}: Batch 0 obs shape = {obs.shape}")
            print(f"Epoch {epoch}: Batch 0 probs shape = {probs.shape}")

        # Forward pass
        logits = model(obs)

        # Log dimension check for predictions
        if logits.shape[1] != action_dim:
            raise ValueError(
                f"Model output dim mismatch: expected {action_dim}, got {logits.shape[1]}"
            )

        loss = soft_cross_entropy(logits, probs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch:02d}] Total Loss = {total_loss:.4f}\n")

# ----------------------
# Save Model
# ----------------------
torch.save(model.state_dict(), "nn_models/base/cfr_bc_policy.pt")
