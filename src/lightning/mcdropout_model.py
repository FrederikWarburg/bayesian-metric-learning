import torch

from src.lightning.base import Base


def enable_dropout(m):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith("Dropout"):
            each_module.train()


class MCDropoutModel(Base):
    def __init__(self, args, savepath, seed):
        super().__init__(args, savepath, seed)

    def forward(self, x, n_samples=1):

        enable_dropout(self.model)

        zs = []
        for i in range(n_samples):
            z = self.model(x)
            zs.append(z["z_mu"])

        zs = torch.stack(zs)

        # compute statistics
        z_mu = zs.mean(dim=0)
        z_sigma = zs.std(dim=0)

        # ensure that we are on unit sphere
        z_mu = z_mu / z_mu.norm(dim=-1, keepdim=True)

        return {"z_mu": z_mu, "z_sigma": z_sigma, "z_samples": zs.permute(1, 0, 2)}

    def compute_loss(self, output, y, indices_tuple):

        # this a hack: pytorch-metric-learning does not use the labels if indices_tuple is provided,
        # however, the shape of the labels are required to be 1D.
        place_holder = torch.zeros(y.size(0), device=y.device)

        loss = self.criterion(output["z_mu"], place_holder, indices_tuple)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
        )

        return loss
