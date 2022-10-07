from turtle import forward
from lightning.base import Base
from losses.pfe_loss import PfeCriterion
import torch
import torch.nn as nn
import os
import sys


class UncertaintyModule(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.backbone = model.backbone
        self.fc_mu = model.linear 

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Freeze backbone parameters
        for param in self.fc_mu.parameters():
            param.requires_grad = False        
        

        in_features = self.fc_mu[0].in_features
        latent_size = self.fc_mu[-2].out_features

        self.fc_log_var = nn.Sequential(
            nn.Linear(in_features, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
            nn.BatchNorm1d(latent_size),
        )

        # Scale and shift parameters from paper
        self.beta = nn.Parameter(torch.zeros(latent_size) - 7, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(latent_size) * 1e-4, requires_grad=True)

    def scale_and_shift(self, x):
        return self.gamma * x + self.beta

    def forward(self, x):

        b, c, h, w = x.shape

        # Non-trainable
        features = self.backbone(x)

        # flatten
        features = features.view(b, -1)

        mu = self.fc_mu(features)

        # Get log var
        log_var = self.fc_log_var(features)

        log_var = self.scale_and_shift(log_var)

        sigma = (log_var * 0.5).exp()

        return {"z_mu" : mu, "z_sigma" : sigma}

def rename_keys(statedict):

    new_dict = {k.replace("model.", "") : statedict[k] for k in statedict.keys()}
    return new_dict

class PfeModel(Base):
    def __init__(self, args):
        super().__init__(args)

        # load model checkpoint
        if not os.path.isfile(args.resume):
            print(f"path {args.resume} not found.")
            print("pfe requires a pretrained model")
            print("fix path and try again.")
            sys.exit()

        model = self.model.load_state_dict(rename_keys(torch.load(args.resume)["state_dict"]))
           

        # encapsolate model in an uncertainty module
        # and freeze deterministic part of model 
        self.model = UncertaintyModule(self.model)

        # overwrite criterion with pfe loss
        self.criterion = PfeCriterion()


    def compute_loss(self, output, y, indices_tuple):

        loss = self.criterion(output["z_mu"], output["z_sigma"], indices_tuple)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
        )

        return loss