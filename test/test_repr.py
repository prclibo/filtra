import unittest
import itertools
import numpy as np

import torch
from torch import nn

from sscnn.conv import *
from sscnn.utils import *

class ReprTest(unittest.TestCase):
    def setUp(self):
        torch.set_printoptions(precision=2, sci_mode=False, linewidth=160)
        np.set_printoptions(precision=2, suppress=True, linewidth=160)
        
        self.quarter = 2
        self.rotation = self.quarter * 4

        torch.manual_seed(20)
        torch.cuda.manual_seed(20)

    def test_regular_mat(self):
        group = (2, self.rotation)
        elem = (1, 1)
        mat = regular_mat(elem, group)
        print(mat)

    def test_comp_dctmat(self):
        group = (2, 6)
        elem = (0, 1)
        dct_mat = comp_dctmat(group)

        H, W = dct_mat.shape
        dct_mat = dct_mat.reshape(H, 2, -1).permute(0, 2, 1).reshape(H, W)
        print(dct_mat)

        irreps = [(s, r) for s in range(group[0]) for r in range(group[1] // 2 + 1)]
        repr_mat = block_diag(irrep_mat(elem, irrep, group) for irrep in irreps)

        select = [(i * 2,) if g1 in [0, group[1] / 2] else (i * 2, i * 2 + 1)\
                for i, (g0, g1) in enumerate(irreps)]
        select = itertools.chain.from_iterable(select)
        select = list(select)

        dct_mat = dct_mat[:, select]
        repr_mat = repr_mat[:, select][select, :]

        regular_mat0 = regular_mat(elem, group)
        regular_mat1 = dct_mat @ repr_mat @ dct_mat.transpose(0, 1)
        
        print(regular_mat0)
        print(regular_mat1)

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = [plt.subplot(1, 4, i + 1) for i in range(4)]
        ax[0].imshow(regular_mat1, cmap=plt.cm.RdYlGn, vmin=-1, vmax=1)
        ax[1].imshow(dct_mat, cmap=plt.cm.RdYlGn, vmin=-1, vmax=1)
        ax[2].imshow(repr_mat.transpose(0, 1), cmap=plt.cm.RdYlGn, vmin=-1, vmax=1)
        ax[3].imshow(dct_mat.transpose(0, 1), cmap=plt.cm.RdYlGn, vmin=-1, vmax=1)
        for a in ax:
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)
            a.set_axis_off()

        plt.show()
        for i, a in enumerate(ax):
            extent = a.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(f'decomp_{elem[0]}_{i}.pdf', bbox_inches=extent)

        import pdb; pdb.set_trace()
        
        

t = ReprTest()
t.setUp()
t.test_comp_dctmat()
# t.test_regular_mat()

