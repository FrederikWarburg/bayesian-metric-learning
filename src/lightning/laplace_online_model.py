from lightning.base import Base
import torch
import torch.nn as nn
import os
import sys

sys.path.append("../stochman")
from stochman import nnj
from stochman.hessian import HessianCalculator
from stochman.laplace import DiagLaplace
from stochman.utils import convert_to_stochman
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm


class LaplaceOnlineModel(Base):
    def __init__(self, args, savepath):
        super().__init__(args, savepath)

        # transfer part of model to stochman
        self.model.linear = convert_to_stochman(self.model.linear)

        self.hessian_calculator = HessianCalculator(wrt="weight",
                                                    loss_func="contrastive",
                                                    shape="diagonal",
                                                    speed="half")

        self.laplace = DiagLaplace()

        self.dataset_size = args.dataset_size
        self.hessian = self.laplace.init_hessian(self.dataset_size, self.model.linear, "cuda:0")
         
        self.hessian_memory_factor = args.hessian_memory_factor


    def training_step(self, batch, batch_idx):

        x, y = self.format_batch(batch)

        x = self.model.backbone(x)

        # get mean and std of posterior
        sigma_q = self.laplace.posterior_scale(torch.relu(self.hessian))
        mu_q = parameters_to_vector(self.model.linear.parameters()).unsqueeze(1)

        # init variables to store running sums
        loss = 0
        hessian = None

        # draw samples from the nn (sample nn)
        samples = self.laplace.sample(mu_q, sigma_q, self.train_n_samples)
        for net_sample in samples:

            # replace the network parameters with the sampled parameters
            vector_to_parameters(net_sample, self.model.linear.parameters())

            z = self.model.linear(x)

            indices_tuple = self.get_indices_tuple(z, y)

            loss += self.compute_loss(z, y, indices_tuple)

            with torch.inference_mode():

                # randomly choose 5000 pairs if more than 5000 pairs available.
                #TODO: decide what to do. What pairs should we use to compute the hessian over?
                # does it matter? What experiments should we run to get a better idea?
                if len(indices_tuple[0]) > 5000:
                    idx = torch.randperm(indices_tuple[0].size(0))[:5000]
                    indices_tuple = (indices_tuple[0][idx], indices_tuple[1][idx], indices_tuple[2][idx])

                h_s = self.hessian_calculator.compute_hessian(x.detach(), self.model.linear, indices_tuple)
                h_s = self.laplace.scale(h_s, x.shape[0], self.dataset_size)

            if hessian is None:
                hessian = h_s
            else:
                hessian += hessian

        # reset the network parameters with the mean parameter (MAP estimate parameters)
        vector_to_parameters(mu_q, self.model.linear.parameters())
        loss = loss / self.train_n_samples
        hessian = hessian / self.train_n_samples

        self.hessian = self.hessian_memory_factor * self.hessian + hessian

        # add images to tensorboard every epoch
        if self.current_epoch == self.counter:
            if self.current_epoch < 5:
                self.log_triplets(x, indices_tuple)
            self.counter += 1

        return loss

    def forward(self, x, n_samples=1):

        x = self.model.backbone(x)

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


    def compute_loss(self, z, y, indices_tuple):

        # this a hack: pytorch-metric-learning does not use the labels if indices_tuple is provided,
        # however, the shape of the labels are required to be 1D.
        place_holder = torch.zeros(y.size(0), device=y.device)

        loss = self.criterion(z, place_holder, indices_tuple)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
        )

        return loss
