import sys
sys.path.append('/mnt/workspace/sscnn')
sys.path.append('/mnt/workspace/sscnn/perf')

import torch
from torch import nn

from sscnn.conv import *
from sscnn.utils import *
import sscnn.e2cnn
from models import C8Backbone, ClassificationHead, RegressionHead

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
kernel_size = 15
batch_size = 1
group = (2, rotation)
height, width = 33, 33
in_irreps = [(1, 0)]
out_mult = 1
conv = IrrepToRegular(group, in_irreps, out_mult, kernel_size, bias=False)
nn.init.uniform_(conv.weight)
conv.weight[0, 0] = torch.eye(kernel_size, kernel_size)
x0 = torch.zeros(batch_size, 2 * len(in_irreps), height, width)
y0 = conv.forward(x0)


fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 8),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )


kernel = conv.filters[:, 0, :, :]
min_, max_ = kernel.min(), kernel.max()
import pdb; pdb.set_trace()

for i, ax in enumerate(grid):
    # ax.imshow(kernel[i].detach(), cmap=plt.cm.RdYlGn, vmin=min_, vmax=max_)
    # ax.imshow(kernel[i].detach(), cmap=plt.cm.gist_rainbow, vmin=min_, vmax=max_)
    ax.imshow(kernel[i].detach(), cmap=plt.cm.gnuplot, vmin=min_, vmax=max_)

plt.show()

import pdb; pdb.set_trace()
