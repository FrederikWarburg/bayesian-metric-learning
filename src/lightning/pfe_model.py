from turtle import forward
from lightning.base import Base
from losses.pfe_loss import PfeCriterion
import torch
import torch.nn as nn
import os
import sys

from models.layers.normalization import GlobalBatchNorm1d


class UncertaintyModule(nn.Module):
    def __init__(self, model, use_global_last_bn_layer=True):
        super().__init__()

        if hasattr(model, "pool"):
            self.backbone = model.backbone
            self.pool = model.pool

            # Freeze backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Freeze pool parameters
            for param in self.pool.parameters():
                param.requires_grad = False
        else:
            self.backbone = model.backbone
        
            # Freeze backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.fc_mu = model.linear

        # Freeze backbone parameters
        for param in self.fc_mu.parameters():
            param.requires_grad = False

        in_features = self.fc_mu[0].in_features
        latent_size = self.fc_mu[-2].out_features

        last_bn_layer = GlobalBatchNorm1d if use_global_last_bn_layer else nn.BatchNorm1d

        self.fc_log_var = nn.Sequential(
            nn.Linear(in_features, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
            last_bn_layer(latent_size),
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
    new_dict = {k.replace("features", "backbone"): new_dict[k] for k in new_dict.keys()}
    return new_dict


class PfeModel(Base):
    def __init__(self, args, savepath, seed):
        super().__init__(args, savepath, seed)

        # load model checkpoint
        resume = os.path.join(args.resume, str(seed), "checkpoints/best.ckpt")
        if not os.path.isfile(resume):
            print(f"path {resume} not found.")
            print("pfe requires a pretrained model")
            print("fix path and try again.")
            sys.exit()

        self.model.load_state_dict(rename_keys(torch.load(resume)["state_dict"]))

        # encapsolate model in an uncertainty module
        # and freeze deterministic part of model
        self.model = UncertaintyModule(self.model, args.get("use_global_last_bn_layer", True))

        # overwrite criterion with pfe loss
        self.criterion = PfeCriterion()

    def forward(self, x, n_samples=1):

        output = self.model(x, n_samples)

        return output

    def compute_loss(self, output, y, indices_tuple):

        loss = self.criterion(output["z_mu"], output["z_sigma"], indices_tuple)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
        )

        return loss
