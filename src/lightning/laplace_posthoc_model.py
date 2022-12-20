import os
import sys

import torch
from stochman import ArccosHessianCalculator, ContrastiveHessianCalculator
from stochman.laplace import DiagLaplace
from stochman.utils import convert_to_stochman
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

from src.lightning.base import Base

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

        self.max_pairs = args.max_pairs

        # load model checkpoint
        resume = os.path.join(args.resume, str(seed), "checkpoints/best.ckpt")
        if not os.path.isfile(resume):
            print(f"path {resume} not found.")
            print("laplace posthoc requires a pretrained model")
            print("fix path and try again.")
            sys.exit()

        # load model parameters
        self.model.load_state_dict(rename_keys(torch.load(resume)["state_dict"]))

        self.hessian_calculator = hessian_calculators[args.loss](
            wrt="weight",
            shape="diagonal",
            speed="half",
            method=args.loss_approx,
        )

        # if arccos, then remove normalization layer from model
        if self.args.loss == "arccos":
            self.model.linear = convert_to_stochman(self.model.linear[:-1])
        else:
            self.model.linear = convert_to_stochman(self.model.linear)

        self.laplace = DiagLaplace()

        # register hessian. It will be overwritten when fitting model
        hessian = torch.zeros_like(parameters_to_vector(self.model.linear.parameters()), device="cuda:0")
        self.register_buffer("hessian", hessian)

    def forward(self, x, n_samples=1):

        x = self.model.backbone(x)
        if hasattr(self.model, "pool"):
            x = self.model.pool(x)

        if not hasattr(self, "hessian"):
            print("==> no hessian found!")
            z = self.model.linear(x)
            return {"z_mu": z}

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

            h = self.hessian_calculator.compute_hessian(x, self.model.linear, indices_tuple)

            self.hessian += h

        savepath = f"{self.savepath.replace('/results', '')}/checkpoints"
        os.makedirs(savepath, exist_ok=True)
        torch.save(self.state_dict(), savepath + "/best.ckpt")
