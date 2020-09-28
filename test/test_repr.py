import unittest
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
        group = (2, 5)
        elem = (1, 2)
        dct_mat = comp_dctmat(group)
        irreps = [(s, r) for s in range(group[0]) for r in range(group[1] // 2 + 1)]
        repr_mat = block_diag(irrep_mat(elem, irrep, group) for irrep in irreps)

        regular_mat0 = regular_mat(elem, group)
        regular_mat1 = dct_mat @ repr_mat @ dct_mat.transpose(0, 1)
        
        print(regular_mat0)
        print(regular_mat1)

        import pdb; pdb.set_trace()

        import matplotlib.pyplot as plt
        plt.imshow(regular_mat1, cmap=plt.cm.RdYlGn, vmin=-1, vmax=1)
        plt.show()
        import pdb; pdb.set_trace()
        
        

t = ReprTest()
t.setUp()
t.test_comp_dctmat()
# t.test_regular_mat()

