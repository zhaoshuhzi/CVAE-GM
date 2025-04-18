import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ProjectionHead(nn.Module):
    def __init__(self, input_dim):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.normalize(self.fc2(x), dim=1)

class CVAE_GM(nn.Module):
    def __init__(self, input_dim, latent_dim=64, num_components=5):
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
        z = z.unsqueeze(1)
        log_probs = -0.5 * torch.sum((z - self.gmm_means) ** 2 / torch.exp(self.gmm_logvars) + self.gmm_logvars, dim=2)
        return -torch.logsumexp(log_probs, dim=1).mean()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        z_proj = self.projection(z)
        return x_recon, mu, logvar, z, z_proj