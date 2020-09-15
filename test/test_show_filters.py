import torch
from torch import nn

from sscnn.conv import *
from sscnn.utils import *

rotation = 8
kernel_size = 8
batch_size = 1
group = (2, rotation)
height, width = 33, 33
in_irreps = [(1, 1)]
out_mult = 1
conv = IrrepToRegular(group, in_irreps, out_mult, kernel_size, bias=False)
nn.init.uniform_(conv.weight)
x0 = torch.zeros(batch_size, 2 * len(in_irreps), height, width)
y0 = conv.forward(x0)

import pdb; pdb.set_trace()
