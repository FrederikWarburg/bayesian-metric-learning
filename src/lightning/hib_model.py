import torch
import torch.nn as nn

from src.lightning.base import Base
from src.losses.hib_loss import HibCriterion
from src.models.layers.normalization import GlobalBatchNorm1d


class UncertaintyModule(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

        self.backbone = model.backbone
        if hasattr(model, "pool"):
            self.pool = model.pool

        self.fc_mu = model.linear

        in_features = self.fc_mu[0].in_features
        latent_size = self.fc_mu[-2].out_features

        self.fc_log_var = nn.Sequential(
            nn.Linear(in_features, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
            GlobalBatchNorm1d(latent_size),
        )

    def forward(self, x, n_samples=1):

        b, c, h, w = x.shape

        # Non-trainable
        if hasattr(self, "pool"):
            features = self.backbone(x)
            features = self.pool(features)
        else:
            features = self.backbone(x)

        mu = self.fc_mu(features)

        # Get log var
        log_var = self.fc_log_var(features)

        sigma = (log_var * 0.5).exp()

        # sample from projected normal distribution
        samples = mu.unsqueeze(1).repeat(1, n_samples, 1) + torch.randn(
            b, n_samples, sigma.shape[-1], device=sigma.device
        ) * sigma.unsqueeze(1).repeat(1, n_samples, 1)
        samples = samples / torch.norm(samples, dim=2, keepdim=True)

        return {"z_mu": mu, "z_sigma": sigma, "z_samples": samples}


def rename_keys(statedict):

    new_dict = {k.replace("model.", ""): statedict[k] for k in statedict.keys()}
    return new_dict


class HibModel(Base):
    def __init__(self, args, savepath, seed):
        super().__init__(args, savepath, seed)

        # encapsolate model in an uncertainty module
        # and freeze deterministic part of model
        self.model = UncertaintyModule(self.model)
        self.train_n_samples = 8

        self.kl_weight = args.get("kl_weight", 1e-6)

        # overwrite criterion with pfe loss
        self.criterion = HibCriterion()

    def forward(self, x, n_samples=1):

        output = self.model(x, n_samples)

        return output

    def compute_loss(self, output, y, indices_tuple):

        loss = self.criterion(
            output["z_samples"], self.model.alpha, self.model.beta, indices_tuple
        )

        # kl divergence
        # kl = log(sigma^2) + (1 + mu^2) / sigma^2
        kl_div = -0.5 * torch.sum(
            torch.log(output["z_sigma"] + 1e-6)
            + (1 + output["z_mu"] ** 2) / (output["z_sigma"] ** 2 + 1e-6)
        )
        loss = loss + self.kl_weight * kl_div

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
        )

        return loss
