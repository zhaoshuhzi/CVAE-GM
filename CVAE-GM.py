import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import KFold
import numpy as np
import random

# Constants
BATCH_SIZE = 128
EPOCHS = 200
LATENT_DIM = 64
NUM_GMM_COMPONENTS = 5
FREQ_BANDS = ['delta', 'slow_theta', 'fast_theta', 'alpha', 'beta', 'gamma']
TAU = 0.07
ALPHA = 0.5
BETA = 1.0
LEARNING_RATE = 1e-3

# Simulated dataset placeholder
class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # shape (N, features)
        self.labels = labels  # SSD subtypes or pseudo-labels for contrastive/self-supervised

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Encoder network
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mean(h), self.fc_logvar(h)

# Decoder network
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

# Projection Head for contrastive learning
class ProjectionHead(nn.Module):
    def __init__(self, input_dim):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.normalize(self.fc2(x), dim=1)

# CVAE-GM Model
class CVAE_GM(nn.Module):
    def __init__(self, input_dim, latent_dim, num_components):
        super(CVAE_GM, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.projection = ProjectionHead(latent_dim)
        self.gmm_means = nn.Parameter(torch.randn(num_components, latent_dim))
        self.gmm_logvars = nn.Parameter(torch.randn(num_components, latent_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def gmm_kl_divergence(self, z):
        # simplified GMM KL divergence (placeholder)
        z = z.unsqueeze(1)
        log_probs = -0.5 * torch.sum((z - self.gmm_means) ** 2 / torch.exp(self.gmm_logvars) + self.gmm_logvars, dim=2)
        return -torch.logsumexp(log_probs, dim=1).mean()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        z_proj = self.projection(z)
        return x_recon, mu, logvar, z, z_proj

# Contrastive Loss
def contrastive_loss(z1, z2, temperature=TAU):
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim_matrix = sim_matrix / temperature

    labels = torch.arange(batch_size).to(z.device)
    labels = torch.cat([labels, labels], dim=0)

    loss = F.cross_entropy(sim_matrix, labels)
    return loss

# Training and Evaluation
def train_model(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for x, _ in dataloader:
        x = x.to(device)
        x1, x2 = random_split_frequencies(x)

        x_recon, mu, logvar, z, z_proj1 = model(x1)
        _, _, _, _, z_proj2 = model(x2)

        recon_loss = F.mse_loss(x_recon, x1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(x)
        gmm_kl = model.gmm_kl_divergence(z)
        cont_loss = contrastive_loss(z_proj1, z_proj2)

        loss = recon_loss + ALPHA * (kl_loss + gmm_kl) + BETA * cont_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step(total_loss)
    return total_loss / len(dataloader)

def random_split_frequencies(x):
    # placeholder: simulate selecting two different frequency combinations
    indices = np.arange(x.shape[1])
    split = np.random.choice(indices, size=x.shape[1] // 2, replace=False)
    other = np.setdiff1d(indices, split)
    return x[:, split], x[:, other]

# Cross-validation training loop
def cross_validate(dataset, input_dim, device):
    kf = KFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold + 1}')
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

        model = CVAE_GM(input_dim=input_dim, latent_dim=LATENT_DIM, num_components=NUM_GMM_COMPONENTS).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(EPOCHS):
            loss = train_model(model, train_loader, optimizer, scheduler, device)
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > 20:
                    print("Early stopping.")
                    break

# Example usage:
# Replace with real data
num_features = 6 * 64  # EEG frequency bands x channels/regions
dummy_data = torch.randn(500, num_features)
dummy_labels = torch.randint(0, 3, (500,))

dataset = EEGDataset(dummy_data, dummy_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cross_validate(dataset, input_dim=num_features, device=device)
