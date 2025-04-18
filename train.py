import torch
from torch.utils.data import DataLoader, Subset
from torch import optim
from sklearn.model_selection import KFold
from models.cvae_gm import CVAE_GM
from utils.loss import contrastive_loss
from utils.scheduler import EarlyStopping
from data.dataset import EEGDataset
import numpy as np
import argparse

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for x, _ in dataloader:
        x = x.to(device)
        x1, x2 = torch.chunk(x, 2, dim=1)

        x_recon, mu, logvar, z, z_proj1 = model(x1)
        _, _, _, _, z_proj2 = model(x2)

        recon_loss = torch.nn.functional.mse_loss(x_recon, x1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(x)
        gmm_kl = model.gmm_kl_divergence(z)
        cont_loss = contrastive_loss(z_proj1, z_proj2)

        loss = recon_loss + 0.5 * (kl_loss + gmm_kl) + 1.0 * cont_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EEGDataset(args.data_path)
    kf = KFold(n_splits=5)

    for fold, (train_idx, _) in enumerate(kf.split(dataset)):
        print(f"Fold {fold+1}")
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=128, shuffle=True)
        model = CVAE_GM(input_dim=dataset[0][0].shape[0]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        early_stopping = EarlyStopping(patience=20)

        for epoch in range(200):
            loss = train(model, train_loader, optimizer, device)
            scheduler.step(loss)
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
            early_stopping.step(loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args)