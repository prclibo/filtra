import math

import torch
from torch import nn
from nn import functional as F

EPS = 1e-3

class MatMulRemapper(nn.Module):
    def __init__(self, grid):
        '''
        Args:
            grid: N x H x W x 2
        '''
        super(self, MatMulRemapper).__init__()
        interp_mat = self.comp_interp_mat(grid)
        interp_mat_t = interp_mat.transpose(1, 2)

        self.register_buffer('grid', grid)
        self.register_buffer('interp_mat', interp_mat)
        self.register_buffer('interp_mat_t', interp_mat_t)
        pass

    def comp_interp_mat(self, grid):
        N, H, W = grid.shape
        # N x H x W
        disk_mask = (grid[:, :, :, 0] >= 0 - EPS).all(0) &\
                    (grid[:, :, :, 0] <= H - 1 + EPS).all(0) &\
                    (grid[:, :, :, 1] >= 0 - EPS).all(0) &\
                    (grid[:, :, :, 1] <= W - 1 + EPS).all(0)
        num_valid = disk_mask.sum()
        
        # N x num_valid x 2
        grid = torch.stack([
            grid[:, disk_mask, 0].clamp(EPS, H - 1 - EPS),
            grid[:, disk_mask, 1].clamp(EPS, W - 1 - EPS),
        ], -1)
        # N x num_valid x 4 x 2
        corners = grid.floor().long().view(N, -1, 1, 2).expand(-1, -1, 4, -1)
        corners += torch.LongTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        # N x num_valid x 4
        corners = corners[:, :, :, 0] * W + corners[:, :, :, 1]
        # N x num_valid x 2
        frac = grid.frac()
        # N x num_valid x 2
        frac0 = torch.stack([1. - frac[:, :, 0], frac[:, :, 0]], -1)
        frac1 = torch.stack([1. - frac[:, :, 1], frac[:, :, 1]], -1)
        # [N x num_valid] x 2 x 2
        coeffs = torch.bmm(frac0.view(-1, 2, 1), frac.view(-1, 1, 2))
        # N x num_valid x 4
        coeffs = coeffs.view(N, num_valid, 4)

        mat_val  = grid.new_zeros(N, num_valid, H * W)
        mat_val.scatter_(2, corners, coeffs)
        # N x [H x W] x [H x W]
        interp_mat = grid.new_zeros(N, H * W, H * W)
        interp_mat[:, disk_mask.flatten(), :] = mat_val
        return interp_mat

    def forward(self, x):
        '''
        Args:
            x: groups x channels x H x W
        '''
        groups, channels, H, W = x.shape
        x = torch.bmm(x, self.interp_mat_t)
        return x

class FilterTransformer(nn.Module):
    def __init__(self, group: Tuple[int, int], size: Tuple[int, int]):
        check_validity(group)
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

        H, W = size
        grid = F.affine_grid(aff.flatten(0, 1), size, False)
        grid = grid * grid.new_tensor([W / 2, H / 2])
        grid += grid.new_tensor([(W - 1) / 2, (H - 1) / 2])

        self.mm_remapper = MatMulRemapper(grid)

