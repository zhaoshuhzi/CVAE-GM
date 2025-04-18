from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import torch

def evaluate(model, dataloader):
    model.eval()
    zs, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            _, _, _, z, _ = model(x)
            zs.append(z.cpu().numpy())
            labels.append(y)
    # Compute clustering metrics here