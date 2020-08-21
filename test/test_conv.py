# python -m unittest discover
import unittest

import torch
from torch import nn

from sscnn.conv import *
from sscnn.utils import *

class ConvTest(unittest.TestCase):
    def setUp(self):
        self.quarter = 2
        self.order = self.quarter * 4
        self.kernel_size = 5
        self.batch_size = 2
        self.height = 9
        self.width = 9
        torch.manual_seed(23)
        torch.cuda.manual_seed(23)

    def test_regular_to_irrep(self):
        in_mult = 2
        out_irreps = [0, 1, 2]
        conv = RegularToIrrep(self.order, in_mult, out_irreps, self.kernel_size, bias=False)
        nn.init.uniform_(conv.weight)

        x0 = torch.rand(self.batch_size, in_mult * self.order, self.height, self.width)
        y0 = conv.forward(x0)

        x1 = rotate_regulars(x0, self.quarter, self.order)
        y1 = conv.forward(x1)
        
        y1_ = rotate_irreps(y0, self.quarter, self.order, out_irreps)

        err = ((y1 - y1_).abs() / y1.abs()).max()
        print('max_rel_err =', err)
        self.assertTrue(y1.allclose(y1_, rtol=2e-2))

    def test_irrep_to_regular(self):
        in_irreps = [0, 1, 2]
        out_mult = 2
        conv = IrrepToRegular(self.order, in_irreps, out_mult, self.kernel_size, bias=False)
        nn.init.uniform_(conv.weight)

        in_dim = len(comp_irreps_expansion_map(self.order, in_irreps)[0])
        x0 = torch.rand(self.batch_size, in_dim, self.height, self.width)
        y0 = conv.forward(x0)

        x1 = rotate_irreps(x0, self.quarter, self.order, in_irreps)
        y1 = conv.forward(x1)
        
        y1_ = rotate_regulars(y0, self.quarter, self.order)

        err = ((y1 - y1_).abs() / y1.abs()).max()
        print('max_rel_err =', err)
        self.assertTrue(y1.allclose(y1_, rtol=2e-2))

