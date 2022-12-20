import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

import src.models.layers.functional as LF

# --------------------------------------
# Normalization layers
# --------------------------------------


class L2Norm(nn.Module):
    def __init__(self, eps=1e-6):
        super(L2Norm, self).__init__()
        self.eps = eps

    def forward(self, x):
        return LF.l2n(x, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + "(" + "eps=" + str(self.eps) + ")"


class PowerLaw(nn.Module):
    def __init__(self, eps=1e-6):
        super(PowerLaw, self).__init__()
        self.eps = eps

    def forward(self, x):
        return LF.powerlaw(x, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + "(" + "eps=" + str(self.eps) + ")"


class GlobalBatchNorm1d(nn.BatchNorm1d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(GlobalBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input: Tensor) -> Tensor:

        # constrain the batchnorm layer to share same gamma and beta
        self._parameters["weight"] = torch.ones_like(
            self._parameters["weight"]
        ) * self._parameters["weight"].mean(dim=0, keepdim=True)
        self._parameters["bias"] = torch.ones_like(
            self._parameters["bias"]
        ) * self._parameters["bias"].mean(dim=0, keepdim=True)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
