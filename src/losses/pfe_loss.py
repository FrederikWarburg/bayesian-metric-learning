import torch
import torch.nn as nn

class PfeCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, sigma, indices_tuple):

        if len(indices_tuple) == 3:
            a, p, _ = indices_tuple
        elif len(indices_tuple) == 4:
            a, p, _, _ = indices_tuple

        sigma2 = sigma ** 2
        loss = torch.norm(mu[a] - mu[p]) / (sigma2[a] + sigma2[p]) + torch.log(sigma2[a] + sigma2[p])

        return loss.mean()