from turtle import forward
from lightning.base import Base
from losses.pfe_loss import PfeCriterion
import torch
import torch.nn as nn
import os
import sys
from copy import deepcopy


def rename_keys(statedict):

    new_dict = {k.replace("model.", ""): statedict[k] for k in statedict.keys()}
    return new_dict


class DeepEnsembleModel(Base):
    def __init__(self, args, savepath, seed):
        super().__init__(args, savepath, seed)

        self.models = []
        for seed in args.seeds:

            # load model checkpoint
            resume = os.path.join(args.resume, str(seed), "checkpoints/best.ckpt")
            if not os.path.isfile(resume):
                print(f"path {resume} not found.")
                print("deep ensemble requires a pretrained model")
                print("fix path and try again.")
                sys.exit()

            self.model.load_state_dict(rename_keys(torch.load(resume)["state_dict"]))
            model= deepcopy(self.model).to("cuda:0")
            self.models.append(model)

    def forward(self, x, n_samples=1):
        
        zs = []
        for model in self.models:
            z = model(x)
            zs.append(z["z_mu"])

        zs = torch.stack(zs)

        # compute statistics
        z_mu = zs.mean(dim=0)
        z_sigma = zs.std(dim=0)

        return {"z_mu": z_mu, "z_sigma": z_sigma, "z_samples" : zs.permute(1, 0, 2)}

