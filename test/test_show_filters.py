import sys
sys.path.append('/mnt/workspace/sscnn')
sys.path.append('/mnt/workspace/sscnn/perf')

import torch
from torch import nn

from sscnn.conv import *
from sscnn.utils import *
import sscnn.e2cnn
from models import Backbone5x5, ClassificationHead, RegressionHead

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

'''
height, width = 33, 33
# conv_func = nn.R2Conv
conv_func = sscnn.e2cnn.SSConv
# conv_func = sscnn.e2cnn.PlainConv
backbone = C8Backbone(out_channels=2, conv_func=conv_func)
head = RegressionHead(backbone.out_type, conv_func)
model = torch.nn.Sequential(backbone, head)

model.load_state_dict(torch.load('/mnt/workspace/sscnn/orient_state.pth'))

x0 = torch.zeros(1, 1, height, width)
y0 = model.forward(x0)



for conv in backbone.modules():
    if isinstance(conv, sscnn.e2cnn.SSConv):
        break

for i in range(5):
    kernel = conv.conv.filters
    Co, Ci, H, W = kernel.shape
    kernel = kernel.view(8, -1, Ci, H, W)
    kernel = kernel[:, i, 0, :, :]
    min_, max_ = kernel.min(), kernel.max()
    
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, 8),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    
    for i, ax in enumerate(grid):
        ax.imshow(kernel[i].detach(), cmap=plt.cm.RdYlGn, vmin=min_, vmax=max_)
        # ax.imshow(kernel[i].detach(), cmap=plt.cm.gist_rainbow, vmin=min_, vmax=max_)
        # ax.imshow(kernel[i].detach(), cmap=plt.cm.gnuplot, vmin=min_, vmax=max_)
    
    plt.show()

import pdb; pdb.set_trace()

quit()
'''

rotation = 8
kernel_size = 5
batch_size = 1
group = (2, rotation)
height, width = 5, 5
in_irreps = [(0, 0), (1, 1)]
out_mult = 1
conv = IrrepToRegular(group, in_irreps, out_mult, kernel_size, bias=False)
nn.init.zeros_(conv.weight)
# conv.weight[0, 0] = torch.eye(kernel_size, kernel_size)
conv.weight[:, :, 1, 0] = 1.0
conv.weight[:, :, 1, 1] = 1.0
conv.weight[:, :, 1, 2] = 1.0
conv.weight[:, :, 1, 3] = 1.0
conv.weight[:, :, 2, 3] = 1.0
conv.weight[:, :, 3, 3] = 1.0
conv.weight[:, :, 2, 2] = 1.0
x0 = torch.zeros(batch_size, 2 * len(in_irreps), height, width)
y0 = conv.forward(x0)

filters = conv.filters.view(group[0], group[1], out_mult, 2, len(in_irreps), kernel_size, kernel_size)
fig = plt.figure(figsize=(30., 160.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 16),  # creates 2x2 grid of axes
                 axes_pad=0.3,  # pad between axes in inch.
                 )


min_, max_ = filters.min(), filters.max()

for i, ax in enumerate(grid):
    j = i if i < group[0] * group[1] else i + group[0] * group[1]
    i_irrep = j // (group[0] * group[1] * 2)
    j = j % (group[0] * group[1] * 2)
    i_comp = j // (group[0] * group[1])
    j = j % (group[0] * group[1])
    i_ref = j // group[1]
    j = j % group[1]
    i_rot = j
    kernel = filters[i_ref, i_rot, 0, i_comp, i_irrep].detach()
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # ax.imshow(kernel, cmap=plt.cm.gnuplot, vmin=min_, vmax=max_)
    ax.imshow(kernel, cmap=plt.cm.RdYlGn, vmin=min_, vmax=max_)
    # ax.imshow(kernel, cmap=plt.cm.gist_rainbow, vmin=min_, vmax=max_)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

# plt.show()
plt.savefig("filters.svg", format="svg")

# import pdb; pdb.set_trace()
