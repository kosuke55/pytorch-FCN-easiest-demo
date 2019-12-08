import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn._reduction as _Reduction
from torch.nn import Module
import warnings


# class Weighted_mse(nn.Module):

#     def __init__(self):
#         super(Weighted_mse, self).__init__()

#     def forward(self, output, target, weight):
#         loss = torch.sum(weight * (output - target) ** 2)
#         return loss


def wmse_loss(input, target, weight, size_average=None, reduce=None, reduction='mean'):

    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.size(), input.size()),
                      stacklevel=2)
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    if target.requires_grad:
        diff = input - target
        diff[np.where(diff > 0)] /= 8.
        # diff *= weight
        ret = diff ** 2
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    else:
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        ret = torch._C._nn.mse_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))
    return ret


class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class WMSELoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(WMSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, weight):
        return wmse_loss(input, target, weight, reduction=self.reduction)
