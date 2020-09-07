from typing import Tuple, Callable, Iterable, List, Any, Dict, Union
import time
import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from .utils import *
from .rotater import *

def comp_irreps_expan_coeffs(group: Tuple[int, int], irreps: torch.Tensor):
    M = len(irreps)
    angles = torch.arange(group[1]) * math.pi * 2 / group[1]
    angles = torch.ger(angles, irreps[:, 1].float()) # group[1] x M
    bases = torch.stack([angles.cos(), angles.sin()], dim=2) # group[1] x M x 2
    assert bases.shape == (group[1], M, 2)
    bases = bases.view(1, group[1], M, 2)

    flip = ((-1) ** irreps[:, 0]).float() # M
    flip = torch.stack((torch.ones(M), flip)) # 2 x M
    flip = flip[:group[0]].view(group[0], 1, M, 1)
    # group[0] x group[1] x M x 2
    return flip.mul(bases)

class IrrepToRegular(nn.Conv2d):
    def __init__(self, group: Tuple[int, int],
            in_irreps: List[Tuple[int, int]], out_mult: int,
            kernel_size: int, **kwargs):

        in_channels = len(in_irreps)
        out_channels = out_mult
        super(IrrepToRegular, self).__init__(
                in_channels, out_channels, kernel_size, **kwargs)

        self.group = group
        [check_validity(None, irrep, group) for irrep in in_irreps]
        self.out_mult = out_mult
        in_irreps = torch.IntTensor(in_irreps)
        # group[0] x group[1] x len(in_irreps) x 2
        expand_coeffs = comp_irreps_expan_coeffs(self.group, in_irreps)

        self.register_buffer('expand_coeffs', expand_coeffs)

        self.filters = None
        self.rotater = FilterRotater(group, self.kernel_size, False)

        # TODO XXX Deal with bias, refering to e2cnn

    def expand_filters(self, weight):
        C_o, C_i, H, W = weight.shape
        order = self.group[0] * self.group[1]
        weight = weight.view(-1, H, W)
        # group[0] x group[1] x [out_mult x len(in_irreps)] x H x W
        steered = self.rotater.forward(weight)
        # group[0] x group[1] x out_mult x 1 x len(in_irreps) x H x W
        steered = steered.view(self.group + (C_o, 1, -1, H, W))
        # group[0] x group[1] x 1 x 2 x len(in_irreps) x 1 x 1
        coeffs = self.expand_coeffs.view(self.group + (1, 2, -1, 1, 1))
        # group[0] x group[1] x out_mult x 2 x len(in_irreps) x H x W
        filters = steered.mul(coeffs)
        return filters.view(order * C_o, 2 * C_i, H, W)

    def forward(self, x):
        self.filters = self.expand_filters(self.weight)
        x = F.conv2d(x, self.filters, self.bias, self.stride,
                self.padding, self.dilation, self.groups)
        return x

