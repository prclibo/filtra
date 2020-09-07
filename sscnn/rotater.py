from typing import Tuple
import math

import torch
from torch import nn
from torch.nn import functional as F

from sscnn.utils import *

EPS = 1e-5
LARGE = 1e5

class MatMulRemapper(nn.Module):
    def __init__(self, grid):
        '''
        Args:
            grid: N x H x W x 2
        '''
        super(MatMulRemapper, self).__init__()
        interp_mat = self.comp_interp_mat(grid)
        interp_mat_t = interp_mat.transpose(1, 2)

        self.register_buffer('grid', grid)
        self.register_buffer('interp_mat', interp_mat)
        self.register_buffer('interp_mat_t', interp_mat_t)
        pass

    def comp_interp_mat(self, grid):
        N, H, W, _ = grid.shape
        # N x H x W
        disk_mask = (grid[:, :, :, 0] >= 0 - EPS).all(0) &\
                    (grid[:, :, :, 0] <= H - 1 + EPS).all(0) &\
                    (grid[:, :, :, 1] >= 0 - EPS).all(0) &\
                    (grid[:, :, :, 1] <= W - 1 + EPS).all(0)
        disk_mask = disk_mask.flatten()
        num_valid = disk_mask.sum()
        
        grid = grid.view(N, -1, 2)
        # N x num_valid x 2
        grid = torch.stack([
            grid[:, disk_mask, 0].clamp(EPS, H - 1 - EPS),
            grid[:, disk_mask, 1].clamp(EPS, W - 1 - EPS),
        ], -1)
        # N x num_valid x 4 x 2
        corners = grid.floor().long().view(N, -1, 1, 2).expand(-1, -1, 4, -1)
        corners = corners + torch.LongTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        # N x num_valid x 4
        corners = corners[:, :, :, 0] * W + corners[:, :, :, 1]
        # N x num_valid x 2
        frac = grid.frac()
        # N x num_valid x 2
        frac0 = torch.stack([1. - frac[:, :, 0], frac[:, :, 0]], -1)
        frac1 = torch.stack([1. - frac[:, :, 1], frac[:, :, 1]], -1)
        # [N x num_valid] x 2 x 2
        coeffs = torch.bmm(frac0.view(-1, 2, 1), frac1.view(-1, 1, 2))
        # N x num_valid x 4
        coeffs = coeffs.view(N, num_valid, 4)

        mat_val  = grid.new_zeros(N, num_valid, H * W)
        mat_val.scatter_(2, corners, coeffs)
        # N x [H x W] x [H x W]
        interp_mat = grid.new_zeros(N, H * W, H * W)
        interp_mat[:, disk_mask, :] = mat_val
        return interp_mat

    def forward(self, x):
        '''
        Args:
            x: groups x channels x H x W
        '''
        # import matplotlib.pyplot as plt
        # plt.imshow(self.interp_mat[1].reshape(25, 25))
        # plt.show()
        groups, channels, H, W = x.shape
        x = x.reshape(groups, channels, -1)
        x0 = x
        x = torch.bmm(x, self.interp_mat_t)
        x = x.view(groups, channels, H, W)
        return x

class GridSampleRemapper(nn.Module):
    def __init__(self, nlized_grid):
        super(GridSampleRemapper, self).__init__()
        disk_mask = (nlized_grid.norm(dim=-1) > 1).unsqueeze(-1)
        nlized_grid = nlized_grid.masked_fill(disk_mask, LARGE)

        self.register_buffer('nlized_grid', nlized_grid)
    def forward(self, x):
        '''
        Args:
            x: groups x channels x H x W
        '''
        x = F.grid_sample(x, self.nlized_grid, align_corners=False, padding_mode='zeros', mode='bilinear')
        return x

class FilterRotater(nn.Module):
    def __init__(self, group: Tuple[int, int], size: Tuple[int, int]):
        super(FilterRotater, self).__init__()
        check_validity(group=group)
        aff = torch.zeros(group + (2, 3)) # reflection x order x 2 x 3
        angles = torch.arange(group[1]) * math.pi * 2 / group[1]
        cos_na, sin_na = angles.cos(), angles.sin()
        aff[0, :, 0, 0] = cos_na
        aff[0, :, 0, 1] = -sin_na
        aff[0, :, 1, 0] = sin_na
        aff[0, :, 1, 1] = cos_na
        if group[0] > 1:
            aff[1, :, 0, 0] = cos_na
            aff[1, :, 0, 1] = sin_na
            aff[1, :, 1, 0] = sin_na
            aff[1, :, 1, 1] = -cos_na

        order = group[0] * group[1]
        H, W = size
        ngrid = F.affine_grid(aff.flatten(0, 1), (order, 1) + size, False)

        grid = ngrid.flip(-1)
        grid = grid * grid.new_tensor([H / 2, W / 2])
        grid += grid.new_tensor([(H - 1) / 2, (W - 1) / 2])

        self.mm_remapper = MatMulRemapper(grid)
        self.gs_remapper = GridSampleRemapper(ngrid)

    def forward(self, x):
        return self.mm_remapper.forward(x)

