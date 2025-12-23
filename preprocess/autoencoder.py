import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from preprocess.param import AE_LATENT_DIM


# Define the Autoencoder class
class AutoEncoder(nn.Module):
    '''
    AutoEncoder model for binary observation tensor data.
    Input: Observation tensor of shape (Batch_size, 72) - values 0 or 1
    Output: Reconstructed observation tensor of the same shape

    To get the latent space representation, use the encode() method.
    '''
    def __init__(self, hidden_dim):
        super(AutoEncoder, self).__init__()
        self.input_dim = 72  # Dimension of the observation tensor
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, hidden_dim),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, self.input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        logits = self.decoder(encoded)
        probs = torch.sigmoid(logits)
        return logits, probs, encoded
    
    def encode(self, x):
        '''
        Encode the input poker tensor into the latent space.
        Args:
            x: Input poker tensor of shape (Batch_size, 6, 4, 13)
        Returns:
            Encoded representation of shape (Batch_size, hidden_dim)
        '''
        return self.encoder(x)

    def train_model(self, dataloader, criterion, optimizer=None, num_epochs=100):
        '''
        Train the AutoEncoder model.
        Args:
            dataloader: DataLoader for training data.
            criterion: Loss function.
            optimizer: Optimizer for training.
            num_epochs: Number of epochs to train.
        '''
        self.train()
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_acc = 0
            num_batches = 0

            for batch in dataloader:
                inputs = batch[0]
                logits, probs, encoded = self(inputs)
                loss = criterion(logits, inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                predicted = (probs >= 0.5).float()
                accuracy = (predicted == inputs).float().mean()
                total_acc += accuracy.item()
                num_batches += 1

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/num_batches:.4f}, Accuracy: {total_acc/num_batches:.4f}")
    
    def visualize_encoded_data(self, data, labels=None, method="t-SNE", num_samples=500):
        """
        Visualize the encoded data in 2D using t-SNE or PCA.
        Args:
            data: Input data (poker tensors).
            labels: Optional labels for coloring the data points.
            method: Dimensionality reduction method ("tsne" or "pca").
            num_samples: Number of samples to visualize.
        """
        # Select a subset of the data for visualization
        data_subset = data[:num_samples]
        labels_subset = None
        if labels is not None:
            labels_subset = labels[:num_samples] 

        self.eval()

        # Pass the data through the encoder
        with torch.no_grad():
            encoded_data = self.encode(data_subset).numpy()

        # Apply dimensionality reduction
        if method == "t-SNE":
            reducer = TSNE(n_components=2, random_state=42)
        elif method == "PCA":
            reducer = PCA(n_components=2)
        else:
            raise ValueError("Invalid method. Choose 't-SNE' or 'PCA'.")

        reduced_data = reducer.fit_transform(encoded_data)

        # Plot the reduced data
        plt.figure(figsize=(8, 6))
        if labels is not None:
            unique_labels = np.unique(labels_subset)
            label_name = ["Call", "Raise", "Fold", "Check"]
            for label in unique_labels:
                indices = labels_subset == label
                plt.scatter(
                    reduced_data[indices, 0],
                    reduced_data[indices, 1],
                    label=label_name[label],
                    s=10
                )
            plt.legend()
        else:
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=10, c="blue")

        # Remove axis labels
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])

        plt.title(f"Encoded Data Visualization ({method})")
        plt.grid(True)
        plt.show()

    def save_model(self, path):
        '''
        Save the model state to the specified path.
        Args:
            path: File path to save the model.
        '''
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        '''
        Load the model state from the specified path.
        Args:
            path: File path to load the model from.
        '''
        self.load_state_dict(torch.load(path))

def label_cfr_data(cfr_dataset):
    '''
    Generate labels for CFR dataset based on the best action.
    Args:
        cfr_dataset: numpy file containing 'obs' and 'probs'.
    Returns:
        labels: Numpy array of shape (num_samples,) with the index of the best action.
    '''
    probs = cfr_dataset['probs']
    labels = np.argmax(probs, axis=1)
    return labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Evaluate AutoEncoder for Poker Observations")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "evaluate"], help='Mode of operation: train or evaluate')
    mode = parser.parse_args().mode

    # Hyperparameters
    hidden_dim = 16  # You can adjust this
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 30
    
    # Prepare DataLoader
    cfr_dataset = np.load("./task_generation/nn_models/base/cfr_bc_dataset.npz")
    observation_data = torch.tensor(cfr_dataset['obs'], dtype=torch.float32)
    labels = torch.tensor(label_cfr_data(cfr_dataset), dtype=torch.long)
    dataset = TensorDataset(observation_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = AutoEncoder(hidden_dim=hidden_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if mode == "train":
        # Train the autoencoder
        model.train_model(dataloader, criterion, optimizer, num_epochs)

        # save the trained model
        model.save_model("./preprocess/models/autoencoder.pt")
    else:
        # Load the trained model weights (if saved previously)
        model.load_model("./preprocess/models/autoencoder.pt")

    # Test the autoencoder
    model.eval()
    test_samples = observation_data[:5].unsqueeze(0)
    logits, probs, encoded = model(test_samples)
    reconstructed = (probs >= 0.5).float()
    print("Original:", test_samples)
    print("Encoded:", encoded)
    print("Reconstructed:", reconstructed)

    # Visualize the encoded data
    model.visualize_encoded_data(data = observation_data, labels=labels, method="t-SNE", num_samples=5000)
