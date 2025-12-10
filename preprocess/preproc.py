import numpy as np
import torch

from preprocess.formatter import CardTensorFormatter
from preprocess.autoencoder import CardAutoEncoder, ActionAutoEncoder

def load_card_autoencoder(hidden_dim=128, ckpt_path="models/card_ae.pth"):
    ae = CardAutoEncoder(hidden_dim=hidden_dim)
    ae.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return ae

def load_action_autoencoder(hidden_dim=128, ckpt_path="models/action_ae.pth"):
    ae = ActionAutoEncoder(hidden_dim=hidden_dim)
    ae.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return ae

class GameStateToTensor:
    def __init__(
        self,
        latent_card_dim,
        latent_action_dim,
        stack_dim
    ):
        self.card_formatter = CardTensorFormatter()

        self.card_encoder = load_card_autoencoder(
            hidden_dim=latent_card_dim,
            ckpt_path="models/card_ae.pth"
        ).eval()
        self.action_encoder = load_action_autoencoder(
            hidden_dim=latent_action_dim,
            ckpt_path="models/action_ae.pth"
        ).eval()

        for m in [self.card_encoder, self.action_encoder]:
            for p in m.parameters():
                p.requires_grad = False

        self.latent_card_dim = latent_card_dim
        self.latent_action_dim = latent_action_dim
        self.stack_dim = stack_dim

        self.total_dim = latent_card_dim + latent_action_dim + stack_dim

    def extract_card_tensor(self, obs):
        raise NotImplementedError
        return self.card_formatter.cards_to_tensor(
            obs["hole_cards"],
            obs["community_cards"]
        )

    def extract_action_tensor(self, obs):
        raise NotImplementedError
        m = obs["num_action_types"]
        rounds = 4
        channels = rounds * 4
        return torch.zeros((channels, 4, m), dtype=torch.float32)

    def extract_stack_vector(self, obs):
        raise NotImplementedError
        pot = obs["pot"]
        stacks = obs["player_stacks"]
        vec = np.array([pot] + list(stacks), dtype=np.float32)

        return vec

    def encode(self, obs):
        card_tensor = self.extract_card_tensor(obs)
        action_tensor = self.extract_action_tensor(obs)
        stack_vec = self.extract_stack_vector(obs)

        with torch.no_grad():
            latent_card = (
                self.card_encoder.encode(card_tensor.unsqueeze(0))
                .squeeze(0)
                .cpu()
                .numpy()
            )

            latent_action = (
                self.action_encoder.encode(action_tensor.unsqueeze(0))
                .squeeze(0)
                .cpu()
                .numpy()
            )

        return np.concatenate(
            [latent_card, latent_action, stack_vec]
        ).astype(np.float32)
