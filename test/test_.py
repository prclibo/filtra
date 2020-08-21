import numpy as np
import torch
from torch import nn

from sscnn.conv import RegularToIrrep 
from sscnn.utils import *

import e2cnn

np.set_printoptions(suppress=True, linewidth=120)
torch.set_printoptions(linewidth=120)
order = 8
kernel_size = 3
conv = RegularToIrrep(order, 1, [0, 1], kernel_size, bias=False)

print(conv.weight.shape)

nn.init.uniform(conv.weight, -100, 100)

x0 = torch.rand(1, 1, order, 9, 9).flatten(1, 2)
print(x0[0, 0, 0])

y0 = conv.forward(x0)
print(y0[0, 1:, 3, 3])

x1 = rotate_regulars(x0, 2, order)
y1 = conv.forward(x1)
print(y1[0, 1:, 3, 3])
quit()

# x1 = x0.rot90(1, dims=[-2, -1])
# x1 = x1.roll(2, dims=2)
# y1 = conv.forward(x1.flatten(1, 2))
# print(y1[0, 0].rot90(-1, dims=[0, 1]))


g = e2cnn.gspaces.Rot2dOnR2(N=order)
in_type = e2cnn.nn.FieldType(g, [g.regular_repr])
out_type = e2cnn.nn.FieldType(g, [g.trivial_repr])
x0_ = e2cnn.nn.GeometricTensor(x0, in_type)
conv_ = e2cnn.nn.R2Conv(in_type, out_type, kernel_size=kernel_size, bias=False)
y0_ = conv_.forward(x0_)
print(y0_)

x1_ = e2cnn.nn.GeometricTensor(x1, in_type)
y1_ = conv_.forward(x1_)
print(y1_)
