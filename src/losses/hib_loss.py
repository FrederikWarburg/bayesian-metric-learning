import torch
import torch.nn as nn


class HibCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_samples, alpha, beta, indices_tuple):
        
        n_samples = z_samples.shape[1]
        
        if len(indices_tuple) == 3:
            a, p, n = indices_tuple
            ap = an = a
        elif len(indices_tuple) == 4:
            ap, p, an, n = indices_tuple

        loss = 0
        for i in range(n_samples):
            z_i = z_samples[:, i, :]
            for j in range(n_samples):
                z_j = z_samples[:, j, :]
                
                prob_pos = torch.sigmoid(alpha * torch.sum((z_i[ap] - z_j[p])**2, dim=1) + beta)
                prob_neg = torch.sigmoid(alpha * torch.sum((z_i[an] - z_j[n])**2, dim=1) + beta)

                # maximize the probability of positive pairs and minimize the probability of negative pairs
                loss += (-prob_pos + prob_neg).mean()
        
        loss = loss / (n_samples ** 2)

        return loss.mean()
