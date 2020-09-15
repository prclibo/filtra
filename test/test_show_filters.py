import torch
from torch import nn

from sscnn.conv import *
from sscnn.utils import *

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

rotation = 8
kernel_size = 7
batch_size = 1
group = (2, rotation)
height, width = 33, 33
in_irreps = [(1, 0)]
out_mult = 1
conv = IrrepToRegular(group, in_irreps, out_mult, kernel_size, bias=False)
nn.init.uniform_(conv.weight)
x0 = torch.zeros(batch_size, 2 * len(in_irreps), height, width)
y0 = conv.forward(x0)


fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 8),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )


kernel = conv.filters[:, 0, :, :]
min_, max_ = kernel.min(), kernel.max()

for i, ax in enumerate(grid):
    ax.imshow(kernel[i].detach(), cmap=plt.cm.RdYlGn, vmin=min_, vmax=max_)
    # ax.imshow(kernel[i].detach(), cmap=plt.cm.gist_rainbow, vmin=min_, vmax=max_)
    # ax.imshow(kernel[i].detach(), cmap=plt.cm.gnuplot, vmin=min_, vmax=max_)

plt.show()

import pdb; pdb.set_trace()
