# python -m unittest discover
import unittest
import numpy as np

import torch
from torch import nn

from sscnn.conv import *
from sscnn.utils import *

import e2cnn

def unflatten(x, old_dim: int, new_dims: Tuple[int]):
    old_shape = x.shape
    new_shape = old_shape[:old_dim] + new_dims + old_shape[old_dim + 1:]
    return x.view(new_shape)

def mean_ratio(x, y, dim, eps=1e-3):
    mask = (y.abs() < eps) & (x.abs() < eps)
    ratio = x / y
    ratio[mask] = 0
    return ratio.sum(dim) / (~mask).float().sum(dim)

def masked_mean(x, mask, dim):
    x[mask] = 0
    return x.sum(dim) / (~mask).float().sum(dim)

def e2cnn_regular_to_irrep(group, in_mult, out_irreps, kernel_size, bias):
    if group[0] == 1:
        g = e2cnn.gspaces.Rot2dOnR2(N=group[1])
    elif group[1] == 1:
        g = e2cnn.gspaces.Flip2dOnR2(axis=0.)
    else:
        g = e2cnn.gspaces.FlipRot2dOnR2(N=group[1], axis=0.)

    in_type = e2cnn.nn.FieldType(g, [g.regular_repr] * in_mult)
    if group[0] == 1:
        out_type = e2cnn.nn.FieldType(g, [g.irrep(k) for (_, k) in out_irreps])
    elif group[1] == 1:
        raise NotImplementedError
    else:
        out_type = e2cnn.nn.FieldType(g, [g.irrep(j, k) for (j, k) in out_irreps])

    conv = e2cnn.nn.R2Conv(in_type, out_type, kernel_size=kernel_size, bias=bias)
    conv.eval()
    return in_type, out_type, conv

def comp_vec_rel_err(group, irreps, y1, y2):
    '''
    Args:
        y1: N x [2 x len(irreps)] x H x W
    '''
    eps = 1e-3
    N, M, H, W = y1.shape
    y1 = y1.view(N, 2, -1, H, W)
    y2 = y2.view(N, 2, -1, H, W)

    diff = (y1 - y2).norm(None, 1)
    leng1 = y1.norm(None, 1)
    leng2 = y2.norm(None, 1)
    rel_err = masked_mean(diff / leng1, (leng1 < eps) | (leng2 < eps), [0, 2, 3])
    return rel_err


def comp_irrep_rel_err(group, irreps, y1, y2):
    eps = 1e-3
    assert y1.shape == y2.shape
    expan_inds = comp_irreps_expan_inds(group, torch.Tensor(irreps))[0]
    expan_inds = torch.LongTensor(expan_inds)[(None,) + (...,) + (None,) * (y1.dim() - 2)].expand_as(y1)

    sq_diff = torch.zeros((y1.shape[0], len(irreps)) + y1.shape[2:])
    sq_diff.scatter_add_(1, expan_inds, (y1 - y2).square())
    diff = sq_diff.sqrt().flatten(2, -1)

    sq_leng1 = torch.zeros_like(sq_diff)
    sq_leng1.scatter_add_(1, expan_inds, y1.square())
    leng1 = sq_leng1.sqrt().flatten(2, -1)
    sq_leng2 = torch.zeros_like(sq_diff)
    sq_leng2.scatter_add_(1, expan_inds, y2.square())
    leng2 = sq_leng2.sqrt().flatten(2, -1)
    rel_err = masked_mean(diff / leng1, (leng1 < eps) | (leng2 < eps), [0, 2])

    return rel_err

def comp_regular_rel_err(group, y1, y2):
    eps = 1e-3
    assert y1.shape == y2.shape
    N, C, H, W = y1.shape
    order = group[0] * group[1]
    y1 = y1.view(N, order, -1, H, W)
    y2 = y2.view(N, order, -1, H, W)
    diff = (y1 - y2).norm(None, 1)
    leng1 = y1.norm(None, 1)
    leng2 = y2.norm(None, 1)
    rel_err = masked_mean(diff / leng1, (leng1 < eps) | (leng2 < eps), [0, 2, 3])
    return rel_err

def shrink_regular_irrep_kernel(group, irreps, kernel, reverse=False):
    if reverse:
        kernel = kernel.permute(1, 0, 2, 3)
    kernel = unflatten(kernel, 1, (-1,) + group) # dim(irreps) x mult x group[0] x group[1] x H x W
    expan_inds = comp_irreps_expan_inds(group, torch.Tensor(irreps))[0]
    expan_inds = torch.LongTensor(expan_inds)[(...,) + (None,) * 5].expand_as(kernel)
    sq_ker = torch.zeros((len(irreps),) + kernel.shape[1:])
    sq_ker.scatter_add_(0, expan_inds, kernel.square())
    kernel = sq_ker.sqrt().flatten(1, 3)
    if reverse:
        kernel = kernel.permute(1, 0, 2, 3)
    return kernel

class ConvTest(unittest.TestCase):
    def setUp(self):
        torch.set_printoptions(sci_mode=False, linewidth=160)
        
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
        self.center_mask = ~self.border_mask

        self.kernel_patch = self.patch[12:21, 12:21]

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
            print(elem, 'max_rel_err =', err.detach(), 'max_abs_err', (y1 - y1_).abs().max().detach())
            self.assertTrue(y1.allclose(y1_, rtol=2e-2))

    def test_irrep_to_regular_reflection(self):
        out_mult = 1
        group = (2, 1)
        in_irreps = [(0, 0), (1, 0)]
        conv = IrrepToRegular(group, in_irreps, out_mult, self.kernel_size, bias=False)
        nn.init.uniform_(conv.weight)
        for i in [1]: # range(2):
            elem = (i, 0)

            x0 = torch.rand(self.batch_size, 2 * len(in_irreps), self.height, self.width)
            y0 = conv.forward(x0)
            x1 = rotate_irreps(x0, elem, in_irreps, group)
            y1 = conv.forward(x1)
            y1_ = rotate_regulars(y0, elem, group)

            rel_err_ss = comp_regular_rel_err(group, y1, y1_)
            print(rel_err_ss)


    def test_regular_to_irrep_Cn(self):
        in_mult = 2
        group = (1, self.rotation)
        out_irreps = [(s, r) for s in range(group[0]) for r in range(group[1] // 2 + 1)]

        in_type, out_type, r2_conv = e2cnn_regular_to_irrep(group, in_mult, out_irreps, self.kernel_size, bias=False)
        ss_conv = RegularToIrrep(group, in_mult, out_irreps, self.kernel_size, bias=False)

        nn.init.uniform_(ss_conv.weight)

        x0 = torch.zeros(self.batch_size, self.rotation, in_mult, self.height, self.width)
        x0[:, 0, :] = self.patch

        x0 = x0.flatten(1, 2)
        y0_ss = ss_conv.forward(x0)
        y0_r2 = r2_conv.forward(e2cnn.nn.GeometricTensor(x0, in_type)).tensor
        # import pdb; pdb.set_trace()

        for i in range(self.rotation):
            elem = (0, i)

            x1 = rotate_regulars(x0, elem, group)
            y1_ss = ss_conv.forward(x1)
            y2_ss = rotate_irreps(y0_ss, elem, out_irreps, group)

            rel_err_ss = comp_vec_rel_err(group, out_irreps, y1_ss, y2_ss)
            print(rel_err_ss)

            # y1_r2 = r2_conv.forward(e2cnn.nn.GeometricTensor(x1, in_type)).tensor
            # y2_r2 = rotate_irreps(y0_r2, elem, out_irreps, group)
            # rel_err_r2 = comp_vec_rel_err(group, out_irreps, y1_r2, y2_r2)
            # print(rel_err_r2)
            # print('-----------')
            # import pdb; pdb.set_trace()

    def test_regular_to_regular_Cn(self):
        in_mult = 2
        group = (1, self.rotation)
        order = group[0] * group[1]
        out_mult = 3

        ss_conv = RegularToRegular(group, in_mult, out_mult, self.kernel_size, bias=False)
        nn.init.uniform_(ss_conv.weight)
        x0 = torch.zeros(self.batch_size, self.rotation, in_mult, self.height, self.width)
        x0[:, 0, :] = self.patch

        x0 = x0.flatten(1, 2)
        y0_ss = ss_conv.forward(x0)

        for i in range(self.rotation):
            elem = (0, i)

            x1 = rotate_regulars(x0, elem, group)
            y1_ss = ss_conv.forward(x1)
            y2_ss = rotate_regulars(y0_ss, elem, group)

            rel_err_ss = comp_regular_rel_err(group, y1_ss, y2_ss)
            # import pdb; pdb.set_trace()
            print(rel_err_ss)

    def test_irrep_to_regular_Cn(self):
        group = (1, self.rotation)
        in_irreps = [(s, r) for s in range(group[0]) for r in range(group[1] // 2 + 1)]
        in_irreps = [(0, 0)]
        out_mult = 2
        out_mult = 24
        self.kernel_size = 5
        self.batch_size = 1
        conv = IrrepToRegular(group, in_irreps, out_mult, self.kernel_size, bias=False)
        nn.init.uniform_(conv.weight)
        x0 = torch.rand(self.batch_size, 2 * len(in_irreps), self.height, self.width)

        from collections import OrderedDict
        import sscnn.e2cnn
        import e2cnn
        from models import C8Backbone, RegressionHead
        device = 'cpu'
        conv_func = sscnn.e2cnn.SSConv
        conv_func = e2cnn.nn.R2Conv
        # conv_func = sscnn.e2cnn.PlainConv
        backbone = C8Backbone(out_channels=2, conv_func=conv_func)
        head = RegressionHead(backbone.out_type, conv_func)
        model = e2cnn.nn.SequentialModule(OrderedDict([
            ('backbone', backbone), ('head', head)
        ]))
        model = model.to(device)
        model = model.export()
        
        # sdict = torch.load(f'/mnt/workspace/sscnn/orient_state_{conv_func.__name__}.pth')
        sdict = torch.load(f'/home/li/workspace/sscnn/orient_state_{conv_func.__name__}.pth')
        model.load_state_dict(sdict)
        
        # model = model.backbone.block1
        # modules = [_ for _ in model.modules()]
        # assert isinstance(modules[3], IrrepToRegular)
        # conv.weight[:] = modules[3].weight[:]
        conv = torch.nn.Sequential(model.backbone.block1,
                model.backbone.block2,
                # model.backbone.pool1,
                model.backbone.block3,
                # model.backbone.block4,
                # model.backbone.pool2,
                # model.backbone.block5,
                # model.backbone.block6,
                # model.backbone.pool3,
        )
        # conv = model.backbone.block1
        x0 = torch.rand(self.batch_size, 1, self.height, self.width)
        

        for i in range(self.rotation):
            elem = (0, i)

            y0 = conv.forward(x0)
            x1 = rotate_trivials(x0, elem, group)
            # x1 = rotate_irreps(x0, elem, in_irreps, group)
            y1 = conv.forward(x1)
            # import pdb; pdb.set_trace()
            y1_ = rotate_regulars(y0, elem, group)

            y0 = y0.view(self.batch_size, -1, 8, y0.shape[-2], y0.shape[-1])
            y1 = y1.view(self.batch_size, -1, 8, y1.shape[-2], y1.shape[-1])

            print(y0[0, 0, :, y0.shape[-2] // 2, y0.shape[-1] // 2])
            print(y1[0, 0, :, y0.shape[-2] // 2, y0.shape[-1] // 2])
            # print(y1_[0, ::24, y0.shape[-2] // 2, y0.shape[-1] // 2])
            print('--;')

            # rel_err_ss = comp_regular_rel_err(group, y1, y1_)
            # print(elem, 'max_rel_err =', rel_err_ss.detach())

    def test_irrep_to_regular_Dn(self):
        group = (2, self.rotation)
        in_irreps = [(1, 0), (1, 1), (1, 2)]
        out_mult = 3
        conv = IrrepToRegular(group, in_irreps, out_mult, self.kernel_size, bias=False)
        nn.init.uniform_(conv.weight)

        x0 = torch.zeros(self.batch_size, 2 * len(in_irreps), self.height, self.width)
        x0[:, 0] = self.patch
        for i in range(self.rotation):
            elem = (1, i)

            y0 = conv.forward(x0)
            x1 = rotate_irreps(x0, elem, in_irreps, group)
            y1 = conv.forward(x1)
            y1_ = rotate_regulars(y0, elem, group)

            rel_err_ss = comp_regular_rel_err(group, y1, y1_)
            print(elem, 'max_rel_err =', rel_err_ss.detach())
            # import pdb; pdb.set_trace()

    def test_regular_to_irrep_Dn(self):
        in_mult = 1
        group = (2, self.rotation)
        out_irreps = [(1, 0), (1, 1), (1, 2)]

        in_type, out_type, r2_conv = e2cnn_regular_to_irrep(group, in_mult, out_irreps, self.kernel_size, bias=False)
        ss_conv = RegularToIrrep(group, in_mult, out_irreps, self.kernel_size, bias=False)

        nn.init.uniform_(ss_conv.weight)

        x0 = torch.zeros(self.batch_size, 2 * self.rotation, in_mult, self.height, self.width)
        x0[:, 0, :] = self.patch

        x0 = x0.flatten(1, 2)
        y0_ss = ss_conv.forward(x0)

        # import matplotlib.pyplot as plt
        # plt.imshow(torch.cat((x0[0, 0],)).detach())
        # plt.show()
        # import pdb; pdb.set_trace()
        y0_r2 = r2_conv.forward(e2cnn.nn.GeometricTensor(x0, in_type)).tensor

        for i in range(self.rotation):
            elem = (1, i)

            x1 = rotate_regulars(x0, elem, group)
            # import matplotlib.pyplot as plt
            # plt.imshow(torch.cat((x0[0, 0], x1[0, 10])).detach())
            # plt.show()

            y1_ss = ss_conv.forward(x1)
            y2_ss = rotate_irreps(y0_ss, elem, out_irreps, group)
            rel_err_ss = comp_vec_rel_err(group, out_irreps, y1_ss, y2_ss)
            print(rel_err_ss.detach())

            # y1_r2 = r2_conv.forward(e2cnn.nn.GeometricTensor(x1, in_type)).tensor
            # y2_r2 = rotate_irreps(y0_r2, elem, out_irreps, group)
            # rel_err_r2 = comp_vec_rel_err(group, out_irreps, y1_r2, y2_r2)
            # print(rel_err_ss.detach(), rel_err_r2.detach())
            # print('-----------')

    def test_regular_to_regular_Dn(self):
        in_mult = 2
        group = (2, self.rotation)
        order = group[0] * group[1]
        out_mult = 3

        ss_conv = RegularToRegular(group, in_mult, out_mult, self.kernel_size, bias=False)
        nn.init.uniform_(ss_conv.weight)
        x0 = torch.zeros(self.batch_size, order, in_mult, self.height, self.width)
        x0[:, 0, :] = self.patch

        x0 = x0.flatten(1, 2)
        y0_ss = ss_conv.forward(x0)

        for i in range(self.rotation):
            elem = (0, i)

            x1 = rotate_regulars(x0, elem, group)
            y1_ss = ss_conv.forward(x1)
            y2_ss = rotate_regulars(y0_ss, elem, group)

            rel_err_ss = comp_regular_rel_err(group, y1_ss, y2_ss)
            # import pdb; pdb.set_trace()
            print(rel_err_ss)

T = ConvTest()
T.setUp()
# T.test_regular_to_irrep_reflection()
# T.test_irrep_to_regular_reflection()
# T.test_regular_to_irrep_Cn()
# T.test_irrep_to_regular_Cn()
# T.test_irrep_to_regular_Dn()
T.test_regular_to_regular_Cn()
# T.test_regular_to_irrep_Dn()
# T.test_regular_to_regular_Dn()
