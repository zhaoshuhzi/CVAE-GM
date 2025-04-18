import torch
import torch.nn.functional as F

def contrastive_loss(z1, z2, temperature=0.07):
    z = torch.cat([z1, z2], dim=0)
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim_matrix /= temperature

    batch_size = z1.size(0)
    labels = torch.cat([torch.arange(batch_size)] * 2).to(z.device)

    loss = F.cross_entropy(sim_matrix, labels)
    return loss