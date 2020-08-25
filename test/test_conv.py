# python -m unittest discover
import unittest
import numpy as np

import torch
from torch import nn

from sscnn.conv import *
from sscnn.utils import *

import e2cnn

def e2cnn_regular_to_irrep(group, in_mult, out_irreps, kernel_size, bias):
    if group[0] == 1:
        g = e2cnn.gspaces.Rot2dOnR2(N=group[1])
    elif group[1] == 1:
        g = e2cnn.Flip2dOnR2(axis=0)
    else:
        g = e2cnn.FlipRot2dOnR2(N=group[1], axis=0)

    in_type = [g.regular_repr] * in_mult
    if group[0] == 1:
        out_type = [g.irrep(k) for (_, k) in out_irreps]
    elif group[1] == 1:
        raise NotImplementedError
    else:
        raise NotImplementedError

    conv = e2cnn.nn.R2Conv(in_type, out_type, kernel_size=kernel_size, bias=bias)
    return conv

class ConvTest(unittest.TestCase):
    def setUp(self):
        torch.set_printoptions(sci_mode=False)
        
        self.quarter = 2
        self.rotation = self.quarter * 4
        self.kernel_size = 9
        self.batch_size = 2
        self.patch = torch.from_numpy(np.load('test/patch.npy'))
        self.height, self.width = self.patch.shape
        torch.manual_seed(20)
        torch.cuda.manual_seed(20)

        oh, ow = self.height - (self.kernel_size - 1), self.width - (self.kernel_size - 1)
        ph, pw = oh // 4, ow // 4
        self.border_mask = torch.ones(oh, ow).bool()
        # self.border_mask[oh // 4:-oh // 4 + 1, ow // 4:-ow // 4 + 1] = False
        self.border_mask[ph:-ph + 1, pw:-pw + 1] = False

    def test_regular_to_irrep_reflection(self):
        in_mult = 1
        group = (2, 1)
        out_irreps = [(0, 0), (1, 0)]
        conv = RegularToIrrep(group, in_mult, out_irreps, self.kernel_size, bias=False)
        nn.init.uniform_(conv.weight)

        for i in range(2):
            elem = (i, 0)
            x0 = torch.rand(self.batch_size, in_mult * 2, self.height, self.width)
            y0 = conv.forward(x0)

            x1 = rotate_regulars(x0, elem, group)
            y1 = conv.forward(x1)

            y1_ = rotate_irreps(y0, elem, out_irreps, group)

            err = ((y1 - y1_).abs() / y1.abs()).max()
            print(elem, 'max_rel_err =', err)
            self.assertTrue(y1.allclose(y1_, rtol=2e-2))

    def test_irrep_to_regular_reflection(self):
        out_mult = 1
        group = (2, 1)
        in_irreps = [(0, 0), (1, 0)]
        in_dim = len(comp_irreps_expansion_map(group, torch.Tensor(in_irreps))[0])
        conv = IrrepToRegular(group, in_irreps, out_mult, self.kernel_size, bias=False)
        nn.init.uniform_(conv.weight)
        for i in range(2):
            elem = (i, 0)

            x0 = torch.rand(self.batch_size, in_dim, self.height, self.width)
            y0 = conv.forward(x0)
            x1 = rotate_irreps(x0, elem, in_irreps, group)
            y1 = conv.forward(x1)
            y1_ = rotate_regulars(y0, elem, group)

            err = ((y1 - y1_).abs() / y1.abs()).max()
            print(elem, 'max_rel_err =', err, 'max_abs_err', (y1 - y1_).abs().max())
            self.assertTrue(y1.allclose(y1_, rtol=2e-2))

    def test_regular_to_irrep_SO2(self):
        in_mult = 2
        group = (1, self.rotation)
        out_irreps = [(0, 0), (0, 1), (0, 2)]
        conv = RegularToIrrep(group, in_mult, out_irreps, self.kernel_size, bias=False)
        nn.init.normal_(conv.weight)
        nn.init.uniform_(conv.weight)

        x0 = torch.zeros(self.batch_size, in_mult, self.rotation, self.height, self.width)
        # x0 = torch.rand(self.batch_size, in_mult, self.rotation, self.height, self.width)
        x0[:, :, 0] = self.patch
        # x0 = torch.ones(self.batch_size, in_mult, self.rotation, self.height, self.width)

        x0 = x0.flatten(1, 2)
        y0 = conv.forward(x0)

        expan_inds = comp_irreps_expansion_map(group, torch.Tensor(out_irreps))[0]
        expan_inds = torch.LongTensor(expan_inds)[(None,) + (...,) + (None, None)].expand_as(y0)
        for i in range(self.rotation):
            elem = (0, i)

            x1 = rotate_regulars(x0, elem, group)
            y1 = conv.forward(x1)
            y1[:, :, self.border_mask] = y0[:, 2].mean()
            y1_ = rotate_irreps(y0, elem, out_irreps, group)
            y1_[:, :, self.border_mask] = y0[:, 2].mean()

            sq_diff = torch.zeros((self.batch_size, len(out_irreps)) + y0.shape[-2:])
            sq_diff.scatter_add_(1, expan_inds, (y1 - y1_).square())
            diff = sq_diff.sqrt().mean([0, 2, 3])

            sq_leng = torch.zeros_like(sq_diff)
            sq_leng.scatter_add_(1, expan_inds, y1.square())
            leng = sq_leng.sqrt().mean([0, 2, 3])
            rel_err = diff / leng

            print('rel_err', rel_err)

    def test_irrep_to_regular_SO2(self):
        group = (1, self.rotation)
        in_irreps = [(0, 0), (0, 1), (0, 2)]
        out_mult = 2
        conv = IrrepToRegular(group, in_irreps, out_mult, self.kernel_size, bias=False)
        nn.init.uniform_(conv.weight)
        in_dim = len(comp_irreps_expansion_map(group, torch.Tensor(in_irreps))[0])
        for i in range(4):
            elem = (0, self.quarter * i)

            x0 = torch.rand(self.batch_size, in_dim, self.height, self.width)
            y0 = conv.forward(x0)
            x1 = rotate_irreps(x0, elem, in_irreps, group)
            y1 = conv.forward(x1)
            y1_ = rotate_regulars(y0, elem, group)

            err = ((y1 - y1_).abs() / y1.abs()).max()
            print(elem, 'max_rel_err =', err)
            self.assertTrue(y1.allclose(y1_, rtol=2e-2))

    
# T = ConvTest()
# T.setUp()
# T.test_regular_to_irrep_reflection()
# T.test_irrep_to_regular_reflection()
# T.test_regular_to_irrep_SO2()
# T.test_irrep_to_regular_SO2()
