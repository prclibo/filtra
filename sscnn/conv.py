from typing import Tuple, Callable, Iterable, List, Any, Dict

import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

def comp_irreps_expansion_map(order, freqs):
    cos_out, sin_out, cos_in, sin_in = [], [], [], []
    in_inds = []
    f_inds = []
    index = 0
    for fi, freq in enumerate(freqs):
        in_inds.append(fi)
        f_inds.append(0)
        if not freq in [0, order / 2]:
            in_inds.append(fi)
            f_inds.append(1)

    freqs = freqs if torch.is_tensor(freqs) else torch.tensor(freqs)
    angles = torch.arange(order) * math.pi * 2 / order 
    angles = torch.ger(angles, freqs.float()) # order x len(freqs)
    coeffs = torch.stack([angles.cos(), angles.sin()], dim=2) # order x len(freq) x 2

    return in_inds, f_inds, coeffs

def irreps_expand(in_inds, f_inds, coeffs, x):
    ''' x: order x len(freqs) -> '''
    assert coeffs.shape[0] == x.shape[0]

    x = x[:, in_inds] # order x order x ...
    # https://github.com/pytorch/pytorch/issues/9410
    coeffs = coeffs[:, in_inds, f_inds][(...,) + (None,) * (x.dim() - 2)] # order x order
    return x.mul(coeffs)

def comp_dctmat(order):
    # https://stackoverflow.com/questions/53875821/scipy-generate-nxn-discrete-cosine-matrix
    # from scipy.fftpack import dct
    # self.dctmat = dct(np.eye(self.order), axis=1)
    freqs = torch.arange(math.floor(order / 2) + 1)
    params = comp_irreps_expansion_map(order, freqs)
    return irreps_expand(*params, torch.ones(order, len(freqs)))

def comp_affine_grid(order, kernel_size):
    size = torch.tensor((order, 1) + kernel_size)

    aff = torch.zeros([order, 2, 3])
    angles = torch.arange(order) * math.pi * 2 / order 
    cos_na, sin_na = angles.cos(), angles.sin()
    aff[:, 0, 0] = cos_na
    aff[:, 0, 1] = -sin_na
    aff[:, 1, 0] = sin_na
    aff[:, 1, 1] = cos_na

    return F.affine_grid(aff, size.tolist(), False)
    # https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522/18

class RegularToIrrep(nn.Conv2d):
    def __init__(self, order: int,
            in_mult: int, out_irreps: List, kernel_size: int, **kwargs):
        self.order = order
        self.in_mult = in_mult
        self.out_irreps = torch.Tensor(out_irreps)

        in_channels = in_mult 
        out_channels = len(out_irreps)
        super(RegularToIrrep, self).__init__(
                in_channels, out_channels, kernel_size, **kwargs)

        self.grid = comp_affine_grid(order, self.kernel_size)
        self.dctmat = comp_dctmat(order)
        self.expansion_params = comp_irreps_expansion_map(order, self.out_irreps)

        assert all([f <= order / 2 for f in out_irreps])

        # TODO XXX Deal with bias, refering to e2cnn

    def expand_filters(self, weight):
        # len(out_irreps) x in_mult x h x w => order x [len(out_irreps) x in_mult] x h x w
        weight = weight.flatten(0, 1).unsqueeze(0)
        weight = weight.expand(self.order, -1, -1, -1)

        # filter shape => order x [len(out_irreps) x in_mult] x h x w
        steered = F.grid_sample(weight, self.grid, align_corners=False, padding_mode='border')
        shape = (self.order, self.out_channels, self.in_channels,) + self.kernel_size
        # => order x len(out_irreps) x in_mult x h x w
        steered = steered.reshape(shape)
        # => order (steered) x out_dims x in_mult x h x w
        filters = irreps_expand(*self.expansion_params, steered)
        filters = filters.permute(1, 2, 0, 3, 4).flatten(1, 2)
        return filters

    def forward(self, x):
        filters = self.expand_filters(self.weight)
        return F.conv2d(x, filters, self.bias, self.stride,
                self.padding, self.dilation, self.groups)

class IrrepToRegular(RegularToIrrep):
    def __init__(self, order: int,
            in_irreps: List, out_mult: int, kernel_size: int, **kwargs):
        super(IrrepToRegular, self).__init__(order, out_mult, in_irreps, kernel_size, **kwargs)

    def forward(self, x):
        filters = self.expand_filters(self.weight)
        filters= filters.permute(1, 0, 2, 3)
        return F.conv2d(x, filters, self.bias, self.stride,
                self.padding, self.dilation, self.groups)

