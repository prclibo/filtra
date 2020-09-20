import sys
sys.path.append('/mnt/workspace/sscnn/')

from collections import OrderedDict
import torch
import torch.nn.functional as F

from e2cnn import gspaces
from e2cnn import nn

import e2cnn
import sscnn.e2cnn

import numpy as np

from mnist_rot_dataset import RotatedMNISTDataset 
from models import C8Backbone, ClassificationHead, RegressionHead

import matplotlib.pyplot as plt
from sscnn.utils import *

from mnist_rot_dataset import batch_rotate

torch.set_printoptions(sci_mode=False, linewidth=160)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128

conv_func = nn.R2Conv
conv_func = sscnn.e2cnn.SSConv
# conv_func = sscnn.e2cnn.PlainConv
backbone = C8Backbone(out_channels=2, conv_func=conv_func)
head = RegressionHead(backbone.out_type, conv_func)
model = nn.SequentialModule(OrderedDict([
    ('backbone', backbone), ('head', head)
]))
model = model.to(device)
model = model.export()

# sdict = torch.load(f'/mnt/workspace/sscnn/orient_state_{conv_func.__name__}.pth')
sdict = torch.load(f'/home/li/workspace/sscnn/orient_state_{conv_func.__name__}.pth')
model.load_state_dict(sdict)

model = model.backbone.block1

height, width = 33, 33
x = torch.rand(1, 1, height, width).to(device)
y0 = model(x)#.flatten()
# start = torch.atan2(y0[1], y0[0])

NN = 8
# for i in range(NN):
#     angle = np.pi * 2 / NN * i
#     aff = torch.zeros(1, 2, 3)
#     aff[0, :, :2] = rot_mat(angle)
#     grid = F.affine_grid(aff.to(device), x.shape, False)
#     
#     rotated = F.grid_sample(x, grid, align_corners=False,
#             padding_mode='zeros', mode='bilinear') 
#     
#     y1 = model(rotated).flatten()
#     print((torch.atan2(y1[1], y1[0]) - start) / np.pi * 180)
#     # print(torch.acos(torch.dot(y0, y1) / y0.norm() / y1.norm()))
# 
# print(y0, y1)
# import pdb; pdb.set_trace()
x = torch.rand(1, 1, height, width).to(device).expand(NN, -1, -1, -1)
y0 = model(x[0:1]).flatten()
# start = torch.atan2(y0[1], y0[0])


angles = torch.arange(NN).float() / NN * np.pi * 2
rotated = batch_rotate(x, angles)
y = model(rotated)
import pdb; pdb.set_trace()

# pred_angles = torch.atan2(y[:, 1, 0, 0], y[:, 0, 0, 0])
# pred_angles = (pred_angles + np.pi * 2).fmod(np.pi * 2)

print(y[:, :, 0, 0])
print(pred_angles - start)

import pdb; pdb.set_trace()
