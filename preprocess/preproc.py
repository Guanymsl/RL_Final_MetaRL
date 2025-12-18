import torch

from preprocess.autoencoder import AutoEncoder
from preprocess.param import AE_LATENT_DIM

def loadAutoencoder(hidden_dim=AE_LATENT_DIM, ckpt_path="preprocess/models/autoencoder.pt"):
    ae = AutoEncoder(hidden_dim=hidden_dim)
    ae.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return ae

class GameStateToTensor:
    def __init__(
        self,
        latent_dim=AE_LATENT_DIM,
    ):
        self.encoder = loadAutoencoder(
            hidden_dim=latent_dim,
            ckpt_path="preprocess/models/autoencoder.pt"
        ).eval()

        for p in self.encoder.parameters():
            p.requires_grad = False

    def encode(self, obs):
        with torch.no_grad():
            latent_card = self.encoder.encode(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).cpu().numpy()

        return latent_card
