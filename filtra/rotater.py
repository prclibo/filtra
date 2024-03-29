from typing import Tuple
import math

import torch
from torch import nn
from torch.nn import functional as F

from filtra.utils import *

EPS = 1e-5
LARGE = 1e5

class BMMRemapper(nn.Module):
    def __init__(self, grid, interpolation='bilinear'):
        '''
        Args:
            grid: N x H x W x 2
        '''
        super(BMMRemapper, self).__init__()
        self.interpolation = interpolation

        interp_mat = self.comp_interp_mat(grid)
        interp_mat_t = interp_mat.transpose(1, 2)


        self.register_buffer('grid', grid)
        self.register_buffer('interp_mat', interp_mat)
        self.register_buffer('interp_mat_t', interp_mat_t)

    def comp_interp_mat(self, grid):
        N, H, W, _ = grid.shape
        if H == 1 and W == 1:
            return grid.new_ones(N, 1, 1)
        elif H == 1 or W == 1:
            raise NotImplementedError
        else:
            # H x W, corresponding -1 < nlized_grid < 1
            disk_mask = (grid[:, :, :, 0] >= 0 - 0.5).all(0) &\
                        (grid[:, :, :, 0] <= H - 1 + 0.5).all(0) &\
                        (grid[:, :, :, 1] >= 0 - 0.5).all(0) &\
                        (grid[:, :, :, 1] <= W - 1 + 0.5).all(0)
            disk_mask = disk_mask.flatten()
            num_valid = disk_mask.sum()
            
            grid = grid.view(N, -1, 2)
            # N x num_valid x 2
            grid = torch.stack([
                grid[:, disk_mask, 0].clamp(EPS, H - 1 - EPS),
                grid[:, disk_mask, 1].clamp(EPS, W - 1 - EPS),
            ], -1)

            if self.interpolation == 'nearest':
                corners, coeffs = self.comp_interp_mat_helper_nearest(grid, H, W)
            elif self.interpolation == 'bilinear':
                corners, coeffs = self.comp_interp_mat_helper_bilinear(grid, H, W)
            else:
                raise NotImplementedError

            mat_val  = grid.new_zeros(N, num_valid, H * W + 1)
            mat_val.scatter_(2, corners, coeffs)
            # N x [H x W] x [H x W]
            interp_mat = grid.new_zeros(N, H * W, H * W)
            interp_mat[:, disk_mask, :] = mat_val[:, :, :-1]
            return interp_mat

    def comp_interp_mat_helper_nearest(self, grid, H, W):
        N, num_valid, _ = grid.shape
        # N x num_valid x 1 x 2
        corners = grid.round().long().view(N, -1, 1, 2)

        oob_mask = (corners[:, :, :, 0] < 0) |\
                   (corners[:, :, :, 0] > H - 1) |\
                   (corners[:, :, :, 1] < 0) |\
                   (corners[:, :, :, 1] > W - 1)

        # N x num_valid x 1
        corners = corners[:, :, :, 0] * W + corners[:, :, :, 1]
        # N x num_valid x 1
        coeffs = grid.new_ones(N, num_valid, 1)
        # Equivalent to zero padding
        coeffs[oob_mask] = 0
        corners[oob_mask] = H * W

        return corners, coeffs

    def comp_interp_mat_helper_bilinear(self, grid, H, W):
        N, num_valid, _ = grid.shape
        # N x num_valid x 4 x 2
        corners = grid.floor().long().view(N, -1, 1, 2).expand(-1, -1, 4, -1)
        corners = corners + torch.LongTensor([[0, 0], [0, 1], [1, 0], [1, 1]])

        oob_mask = (corners[:, :, :, 0] < 0) |\
                   (corners[:, :, :, 0] > H - 1) |\
                   (corners[:, :, :, 1] < 0) |\
                   (corners[:, :, :, 1] > W - 1)

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
        # Equivalent to zero padding
        coeffs[oob_mask] = 0
        corners[oob_mask] = H * W

        return corners, coeffs

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
        x = torch.bmm(x, self.interp_mat_t)
        x = x.view(groups, channels, H, W)
        return x

class GridSampleRemapper(nn.Module):
    def __init__(self, nlized_grid, interpolation):
        super(GridSampleRemapper, self).__init__()
        disk_mask = (nlized_grid.norm(dim=-1) > 1).unsqueeze(-1)
        nlized_grid = nlized_grid.masked_fill(disk_mask, LARGE)
        nlized_grid = nlized_grid.flatten(0, 1)
        self.interpolation = interpolation

        self.register_buffer('nlized_grid', nlized_grid)
    def forward(self, x):
        '''
        Args:
            x: groups x channels x H x W
        '''
        x = F.grid_sample(x, self.nlized_grid, align_corners=False, padding_mode='zeros', mode=self.interpolation)
        return x

class FilterRotater(nn.Module):
    def __init__(self, group: Tuple[int, int], size: Tuple[int, int], reuse=True, method='bmm', interpolation='bilinear'):
        super(FilterRotater, self).__init__()
        check_validity(group=group)
        self.group = group

        aff = torch.zeros(group + (2, 3)) # reflection x order x 2 x 3
        angles = torch.arange(group[1]) * math.pi * 2 / group[1]
        cos_na, sin_na = angles.cos(), angles.sin()
        aff[0, :, 0, 0] = cos_na
        aff[0, :, 0, 1] = -sin_na
        aff[0, :, 1, 0] = sin_na
        aff[0, :, 1, 1] = cos_na
        if group[0] > 1:
            aff[1, :, 0, 0] = cos_na
            aff[1, :, 0, 1] = -sin_na
            aff[1, :, 1, 0] = -sin_na
            aff[1, :, 1, 1] = -cos_na

        order = group[0] * group[1]
        H, W = size
        # order x H x W
        ngrid = F.affine_grid(aff.flatten(0, 1), (order, 1) + size, False)
        self.ngrid = ngrid
        ngrid = ngrid.view(group + size + (-1,))

        grid = ngrid.flip(-1)
        grid = grid * grid.new_tensor([H / 2, W / 2])
        grid += grid.new_tensor([(H - 1) / 2, (W - 1) / 2])
        self.grid = grid

        if reuse:
            self.divisor = (
                2 if group[0] % 2 == 0 else 1,
                4 if group[1] % 4 == 0 else 2 if group[1] % 2 == 0 else 1
            )
        else:
            self.divisor = (1, 1)
        portion = tuple(g // d for g, d in zip(group, self.divisor))
        if method == 'bmm':
            self.remapper = BMMRemapper(grid[:portion[0], :portion[1]].flatten(0, 1), interpolation)
        elif method == 'grid_sample':
            self.remapper = GridSampleRemapper(ngrid[:portion[0], :portion[1]], interpolation)
        else:
            raise NotImplementedError

    def forward(self, x):
        '''
        Args:
            x: channels x H x W
        '''
        # return self.mm_remapper.forward(x)
        # return self.gs_remapper.forward(x)
        portion = tuple(g // d for g, d in zip(self.group, self.divisor))
        x = x.unsqueeze(0).unsqueeze(0).expand(portion + (-1, -1, -1))
        part = self.remapper.forward(x.flatten(0, 1))
        # group[0] x group[1] x channels x H x W
        part = part.view(portion + part.shape[1:])
        if self.divisor[1] == 4:
            x = torch.cat([part, part.rot90(1, [3, 4]), part.rot90(2, [3, 4]), part.rot90(3, [3, 4])], 1)
        elif self.divisor[1] == 2:
            x = torch.cat([part, part.rot90(2, [3, 4])], 1)
        else:
            x = part
        if self.divisor[0] == 2:
            x = torch.cat([x, x.flip(-2)], 0)

        return x

