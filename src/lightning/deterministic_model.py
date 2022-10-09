from lightning.base import Base
import torch


class DeterministicModel(Base):
    def __init__(self, args):
        super().__init__(args)

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
