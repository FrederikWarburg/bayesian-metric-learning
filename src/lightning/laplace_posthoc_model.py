from lightning.base import Base
import torch
import torch.nn as nn
import os
import sys

sys.path.append("../stochman")
from stochman import nnj
from stochman import ContrastiveHessianCalculator, ArccosHessianCalculator
from stochman.laplace import DiagLaplace, optimize_prior_precision
from stochman.utils import convert_to_stochman
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt

hessian_calculators = {
    "contrastive": ContrastiveHessianCalculator,
    "arccos": ArccosHessianCalculator,
}


def rename_keys(statedict):
    new_dict = {k.replace("model.", ""): statedict[k] for k in statedict.keys()}
    new_dict = {k.replace("features", "backbone"): new_dict[k] for k in new_dict.keys()}
    return new_dict


class LaplacePosthocModel(Base):
    def __init__(self, args, savepath, seed):
        super().__init__(args, savepath, seed)

        self.max_pairs = args.get("max_pairs", 5000)

        # load model checkpoint
        resume = os.path.join(args.resume, str(seed), "checkpoints/best.ckpt")
        if not os.path.isfile(resume):
           print(f"path {resume} not found.")
           print("laplace posthoc requires a pretrained model")
           print("fix path and try again.")
           sys.exit()

        # load model parameters
        self.model.load_state_dict(rename_keys(torch.load(resume)["state_dict"]))
        loss = args.get("loss", "contrastive")
        loss_approx = args.get("loss_approx", "full")

        self.hessian_calculator = hessian_calculators[loss](
            wrt="weight",
            shape="diagonal",
            speed="half",
            method=loss_approx,
        )

        # if arccos, then remove normalization layer from model
        if loss == "arccos":
            self.model.linear = convert_to_stochman(self.model.linear[:-1])
        else:
            self.model.linear = convert_to_stochman(self.model.linear)

        self.laplace = DiagLaplace()
        prior_prec = torch.tensor(1.0)
        self.register_buffer("prior_prec", prior_prec)

        # register hessian. It will be overwritten when fitting model
        hessian = torch.zeros_like(parameters_to_vector(self.model.linear.parameters()), device="cuda:0")
        self.register_buffer("hessian", hessian)

    def forward(self, x, n_samples=1):

        x = self.model.backbone(x)
        if hasattr(self.model, "pool"):
            x = self.model.pool(x)

        # get mean parameters
        mu_q = parameters_to_vector(self.model.linear.parameters()).cuda()

        # forward n times
        zs = []
        for i in range(n_samples):

            # use sample i that was generated in beginning of evaluation
            net_sample = self.nn_weight_samples[i]

            # replace the network parameters with the sampled parameters
            vector_to_parameters(net_sample, self.model.linear.parameters())
            
            z = self.model.linear(x)

            # ensure that we are on unit sphere
            z = z / z.norm(dim=-1, keepdim=True)

            zs.append(z)

        zs = torch.stack(zs)

        # compute statistics
        z_mu = zs.mean(dim=0)
        z_sigma = zs.std(dim=0)
        
        # put mean parameters back
        vector_to_parameters(mu_q, self.model.linear.parameters())

        return {"z_mu": z_mu, "z_sigma": z_sigma, "z_samples": zs.permute(1, 0, 2)}

    def fit(self, datamodule):

        train_loader = datamodule.train_dataloader()
        self.model.cuda()
        self.model.eval()

        for batch in tqdm(train_loader):
        
            x, y = self.format_batch(batch)

            x = x.cuda()
            y = y.cuda()

            with torch.inference_mode():
                x = self.model.backbone(x)
                if hasattr(self.model, "pool"):
                    x = self.model.pool(x)

                z = self.model.linear(x)

                # ensure that we are on unit sphere
                z = z / z.norm(dim=-1, keepdim=True)

            indices_tuple = self.get_indices_tuple(z, y)

            # randomly choose 5000 pairs if more than 5000 pairs available.
            if len(indices_tuple[0]) > self.max_pairs:
                idx = torch.randperm(indices_tuple[0].size(0))[: self.max_pairs]
                indices_tuple = (
                    indices_tuple[0][idx],
                    indices_tuple[1][idx],
                    indices_tuple[2][idx],
                )
            
            h = self.hessian_calculator.compute_hessian(
                x, self.model.linear, indices_tuple
            )

            self.hessian += h

        savepath = f"{self.savepath.replace('/results', '')}/checkpoints"
        os.makedirs(savepath, exist_ok=True)
        torch.save(self.state_dict(), savepath + "/best.ckpt")

    def sample(self, n_samples):

        if not hasattr(self, "hessian"):
            print("==> no hessian found!")
            sys.exit()

        # get mean and std of posterior
        mu_q = parameters_to_vector(self.model.linear.parameters()).unsqueeze(1)
        sigma_q = self.laplace.posterior_scale(torch.relu(self.hessian), prior_prec=self.prior_prec)

        # draw samples
        self.nn_weight_samples = self.laplace.sample(mu_q, sigma_q, n_samples)

    def on_validation_epoch_start(self):
        self.sample(self.val_n_samples)

    def on_test_epoch_start(self):
        self.sample(self.test_n_samples)

    def optimize_prior_precision(self):
        
        mu_q = parameters_to_vector(self.model.linear.parameters()).cuda()
        hessian = torch.relu(self.hessian).cuda()
        prior_prec = torch.ones(1, device=mu_q.device).cuda()
        self.prior_prec = optimize_prior_precision(mu_q, hessian, prior_prec, 1)

