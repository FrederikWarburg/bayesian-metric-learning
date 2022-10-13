from turtle import forward
from lightning.base import Base
import torch
import torch.nn as nn
import os
import sys

sys.path.append("../stochman")
from stochman import nnj
from stochman.laplace import HessianCalculator
from models.laplace_utils import DiagLaplace
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

def convert_to_stochman(sequential):

    model = []
    for layer in sequential:

        if layer.__class__.__name__ == "Linear":
            stochman_layer = getattr(nnj, layer.__class__.__name__)(layer.in_features, layer.out_features)
        elif layer.__class__.__name__ == "Conv2d":
            stochman_layer = getattr(nnj, layer.__class__.__name__)(layer.in_channel, layer.out_channel, layer.kernel)
        else:
            stochman_layer = getattr(nnj, layer.__class__.__name__)()

        model.append(stochman_layer)

    model = nnj.Sequential(model, add_hooks=True)

    return model


def rename_keys(statedict):

    new_dict = {k.replace("model.", ""): statedict[k] for k in statedict.keys()}
    return new_dict


class LaplacePosthocModel(Base):
    def __init__(self, args):
        super().__init__(args)

        # load model checkpoint
        if not os.path.isfile(args.resume):
            print(f"path {args.resume} not found.")
            print("laplace posthoc requires a pretrained model")
            print("fix path and try again.")
            sys.exit()

        # transfer part of model to stochman
        self.model.linear = convert_to_stochman(self.model.linear)

        # load model parameters
        self.model.load_state_dict(rename_keys(torch.load(args.resume)["state_dict"]))

        self.hessian_calculator = HessianCalculator(wrt="weight",
                                                    loss_func="contrastive",
                                                    shape="diagonal",
                                                    speed="half")

        self.laplace = DiagLaplace()

    def forward(self, x, n_samples=1):

        x = self.model.backbone(x)

        if not hasattr(self, "hessian"):
            print("==> no hessian found!")
            z = self.model.linear(x)
            return {"z_mu" : z}
        
        # get mean and std of posterior
        mu_q = parameters_to_vector(self.model.linear.parameters()).unsqueeze(1)
        self.hessian = torch.relu(self.hessian)
        sigma_q = self.laplace.posterior_scale(self.hessian)

        # draw samples
        samples = self.laplace.sample(mu_q, sigma_q, n_samples)

        # forward n times
        zs = []
        for net_sample in samples:

            # replace the network parameters with the sampled parameters
            vector_to_parameters(net_sample, self.model.linear.parameters())

            z = self.model.linear(x)

            zs.append(z)

        zs = torch.stack(zs)

        # compute statistics
        z_mu = zs.mean(dim=0) 
        z_sigma = zs.std(dim=0)

        # put mean parameters back
        vector_to_parameters(mu_q, self.model.linear.parameters())

        return {"z_mu" : z_mu, "z_sigma" : z_sigma, "z_samples": zs.permute(1,0,2)}

    def fit(self, datamodule):

        train_loader = datamodule.train_dataloader()
        self.model.cuda()
        self.model.eval()

        hessian = None

        for batch in tqdm(train_loader):
            with torch.inference_mode():

                x, y = self.format_batch(batch)

                x = x.cuda()
                y = y.cuda()

                x = self.model.backbone(x)

                z_mu = self.model.linear(x)

                tuple_indices = self.get_indices_tuple(z_mu, y)

                # randomly choose 5000 pairs if more than 5000 pairs available.
                if len(tuple_indices[0]) > 5000:
                    idx = torch.randperm(tuple_indices[0].size(0))[:5000]
                    tuple_indices = (tuple_indices[0][idx], tuple_indices[1][idx], tuple_indices[2][idx])

                h = self.hessian_calculator.compute_hessian(x, self.model.linear, tuple_indices)

                if hessian is None:
                    hessian = h
                else:
                    hessian += h

        self.hessian = hessian

        savepath = f"../lightning_logs/{self.args.dataset}/{self.args.model}/checkpoints"
        os.makedirs(savepath, exist_ok=True)
        torch.save(hessian, savepath + "/hessian.pth")
        torch.save(self.model.state_dict(), savepath + "/best.ckpt")