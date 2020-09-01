from typing import Tuple, Callable, Iterable, List, Any, Dict, Union
import time
import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from .utils import *

def unflatten(x, old_dim: int, new_dims: Tuple[int]):
    old_shape = x.shape
    new_shape = old_shape[:old_dim] + new_dims + old_shape[old_dim + 1:]
    return x.view(new_shape)

def comp_irreps_expan_inds(group: Tuple[int, int], irreps: torch.Tensor):
    in_inds = []
    cs_inds = []
    index = 0
    for i, irrep in enumerate(irreps):
        in_inds.append(i)
        cs_inds.append(0)
        if not irrep[1] in [0, group[1] / 2]:
            in_inds.append(i)
            cs_inds.append(1)

    return torch.LongTensor(in_inds), torch.LongTensor(cs_inds)

def comp_irreps_expan_coeffs(group: Tuple[int, int], irreps: torch.Tensor,
        in_inds, cs_inds):
    angles = torch.arange(group[1]) * math.pi * 2 / group[1]
    angles = torch.ger(angles, irreps[:, 1].float()) # order x len(freqs)
    bases = torch.stack([angles.cos(), angles.sin()], dim=2) # order x len(irreps) x 2
    bases = bases[:, in_inds, cs_inds] # group[1] x dim(irreps)
    flip = ((-1) ** irreps[:, 0])[in_inds].float() # dim(irreps)
    flip = torch.stack((torch.ones(len(in_inds)), flip)) # 2 x dim(irreps)
    flip = flip[:group[0]]
    coeffs = flip.unsqueeze(1).mul(bases.unsqueeze(0)) # 2 x group[1] x dim(irreps)

    return coeffs.detach()

def irreps_expand(in_inds, coeffs, x):
    ''' x: sides x order x len(freqs) x ... -> '''
    assert x.shape[0] in [1, 2] and coeffs.shape[:2] == x.shape[:2]

    # x = x[:, :, torch.LongTensor(in_inds)] # group[0] x group[1] x dim(irreps) x ...
    shape = x.shape
    x = x.index_select(2, in_inds)
    # print(shape, '->', x.shape)
    # https://github.com/pytorch/pytorch/issues/9410
    # coeffs = coeffs[(...,) + (None,) * (x.dim() - 3)]
    coeffs = coeffs.unsqueeze(-1)
    coeffs = unflatten(coeffs, coeffs.dim() - 1, (1,) * (x.dim() - 3))
    x = x.mul(coeffs)

    return x

def comp_dctmat(group):
    order = group[0] * group[1]
    # https://stackoverflow.com/questions/53875821/scipy-generate-nxn-discrete-cosine-matrix
    # from scipy.fftpack import dct
    # self.dctmat = dct(np.eye(self.group[1]), axis=1)
    freqs = torch.arange(math.floor(group[1] / 2) + 1)
    irreps = torch.Tensor([[s, r] for s in range(group[0]) for r in range(group[1] // 2 + 1)])
    in_inds, cs_inds = comp_irreps_expan_inds(group, irreps)
    coeffs = comp_irreps_expan_coeffs(group, irreps, in_inds, cs_inds)
    dct = irreps_expand(in_inds, coeffs, torch.ones(group[0], group[1], len(irreps)))
    return F.normalize(dct.flatten(0, 1), dim=0).detach()

def comp_affine_grid(order, kernel_size):
    size = torch.tensor((2 * order, 1) + kernel_size)

    aff = torch.zeros([2, order, 2, 3]) # reflection x order x 2 x 3
    angles = torch.arange(order) * math.pi * 2 / order 
    cos_na, sin_na = angles.cos(), angles.sin()
    aff[0, :, 0, 0] = cos_na
    aff[0, :, 0, 1] = -sin_na
    aff[0, :, 1, 0] = sin_na
    aff[0, :, 1, 1] = cos_na
    aff[1, :, 0, 0] = cos_na
    aff[1, :, 0, 1] = sin_na
    aff[1, :, 1, 0] = sin_na
    aff[1, :, 1, 1] = -cos_na
    grid = F.affine_grid(aff.flatten(0, 1), size.tolist(), False)

    disk_mask = (grid.norm(dim=-1) > 1).unsqueeze(-1)
    grid.masked_fill_(disk_mask, 100)

    return unflatten(grid, 0, (2, order))

class RegularToIrrep(nn.Conv2d):
    def __init__(self, group: Tuple[int, int],
            in_mult: int, out_irreps: List[Tuple[int, int]],
            kernel_size: int, **kwargs):

        in_channels = in_mult 
        out_channels = len(out_irreps)
        super(RegularToIrrep, self).__init__(
                in_channels, out_channels, kernel_size, **kwargs)

        self.group = group
        [check_validity(None, irrep, group) for irrep in out_irreps]

        self.in_mult = in_mult
        out_irreps = torch.IntTensor(out_irreps)
        grid = comp_affine_grid(self.group[1], self.kernel_size)
        grid = grid[0:self.group[0], 0:self.group[1]].flatten(0, 1)
        in_inds, cs_inds = comp_irreps_expan_inds(self.group, out_irreps)
        expan_coeffs = comp_irreps_expan_coeffs(self.group, out_irreps,
                in_inds, cs_inds)
        self.register_buffer('out_irreps', out_irreps)
        self.register_buffer('grid', grid)
        self.register_buffer('expan_coeffs', expan_coeffs)
        self.register_buffer('in_inds', in_inds)

        H, W = kernel_size, kernel_size
        map_i = torch.arange(H * W).to(self.weight.device).unsqueeze(0).expand(self.group[0] * self.group[1], -1).flatten()
        self.register_buffer('map_i', map_i)

        steered_weight = self.weight.new_zeros(out_channels * in_channels, group[0] * group[1], H, W)
        self.register_buffer('steered_weight', steered_weight)
        self.filters = None

        # TODO XXX Deal with bias, refering to e2cnn

    def expand_filters(self, weight):
        # len(out_irreps) x in_mult x h x w => [sides x order] x [len(out_irreps) x in_mult] x h x w
        _, _, H, W = weight.shape
        # weight = weight.flatten(0, 1).unsqueeze(0)
        weight = weight.view(1, -1, H, W)
        weight = weight.expand(self.group[0] * self.group[1], -1, -1, -1)
        # filter shape => [sides x order] x [len(out_irreps) x in_mult] x h x w
        steered = F.grid_sample(weight, self.grid, align_corners=False, padding_mode='zeros', mode='nearest') # mode='bilinear')
        # => sides x order x len(out_irreps) x in_mult x h x w
        steered = unflatten(steered, 1, (self.out_channels, self.in_channels))
        steered = unflatten(steered, 0, (self.group[0], self.group[1]))
        # => sides x order (steered) x out_dims x in_mult x h x w
        # steered = self.test_reorder(weight)
        filters = irreps_expand(self.in_inds, self.expan_coeffs, steered)
        # => out_dims x [in_mult x sides x order] x h x w
        filters = filters.permute(2, 3, 0, 1, 4, 5).flatten(1, 3)
        return filters

    def test_reorder(self, weight):
        C_out, C_in, H, W = weight.shape
        weight = weight.reshape(C_out * C_in, H * W)
        # [len(out_irreps) x in_mult] x [sides x order x h x w]
        # weight = weight.index_select(1, self.map_i)
        weight = self.steered_weight
        weight = weight.reshape(C_out, C_in, self.group[0], self.group[1], H, W).permute(2, 3, 0, 1, 4, 5)
        return weight

    def forward(self, x):
        # if not torch.is_tensor(self.filters):
        #     H, W = self.weight.shape[2:]
        #     self.filters = self.weight.new_zeros(len(self.in_inds), self.group[0] * self.group[1] * self.in_mult, H, W)
        self.filters = self.expand_filters(self.weight)
        x = F.conv2d(x, self.filters, self.bias, self.stride,
                self.padding, self.dilation, self.groups)
        return x

class IrrepToRegular(RegularToIrrep):
    def __init__(self, group: Tuple[int, int],
            in_irreps: List[Tuple[int, int]], out_mult: int,
            kernel_size: int, **kwargs):
        super(IrrepToRegular, self).__init__(group, out_mult, in_irreps, kernel_size, **kwargs)

    def forward(self, x):
        # if not torch.is_tensor(self.filters):
        #     H, W = self.weight.shape[2:]
        #     self.filters = self.weight.new_zeros(len(self.in_inds), self.group[0] * self.group[1] * self.in_mult, H, W)
        #     self.filters = self.filters.permute(1, 0, 2, 3)
        self.filters = self.expand_filters(self.weight)
        self.filters = self.filters.permute(1, 0, 2, 3)
        x = F.conv2d(x, self.filters, self.bias, self.stride,
                self.padding, self.dilation, self.groups)
        return x

class RegularToRegular(RegularToIrrep):
    def __init__(self, group: Tuple[int, int], in_mult:int, out_mult: int,
            kernel_size: int, **kwargs):
        out_irreps = [(s, r) for s in range(group[0]) for r in range(group[1] // 2 + 1)]
        out_irreps *= out_mult
        super(RegularToRegular, self).__init__(group, in_mult, out_irreps, kernel_size, **kwargs)

        self.out_mult = out_mult
        self.register_buffer('dct_mat', comp_dctmat(group))

    def forward(self, x):
        order = self.group[0] * self.group[1]
        # N x out_dims x Hout x Wout

        # x = super(RegularToRegular, self).forward(x)
        # x = unflatten(x, 1, (self.out_mult, order))
        # x = torch.einsum('cd,nodhw->nochw', self.dct_mat, x)
        # print(self.dct_mat.shape, x.shape)
        # x = x.flatten(1, 2)

        # out_dims x [in_mult x sides x order] x h x w
        # if not torch.is_tensor(self.filters):
        #     H, W = self.weight.shape[2:]
        #     self.filters = self.weight.new_zeros(len(self.in_inds), self.group[0] * self.group[1] * self.in_mult, H, W)
        self.filters = self.expand_filters(self.weight)
        # [out_dims x in_mult] x [group[0] x group[1]] x h x w
        # w = unflatten(self.filters, 1, (self.in_mult, order)).flatten(0, 1)
        # w = F.conv2d(w, self.dct_mat.view(order, order, 1, 1), None)

        # w = unflatten(self.filters, 0, (self.out_mult, order))
        # w = F.conv3d(w, self.dct_mat[(...,) + (None,) * 3])

        # w = unflatten(self.filters, 0, (self.out_mult, order)).flatten(2, 3)
        # w = F.conv2d(w, self.dct_mat[(...,) + (None,) * 2])

        w = unflatten(self.filters, 0, (self.out_mult, order)).flatten(2, -1)
        w = F.conv1d(w, self.dct_mat.unsqueeze(-1))

        w = w.view(self.filters.shape)
        x = F.conv2d(x, w, self.bias, self.stride,
                self.padding, self.dilation, self.groups)

        return x



