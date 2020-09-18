import sys
sys.path.append('/mnt/workspace/sscnn/')

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128

conv_func = nn.R2Conv
conv_func = sscnn.e2cnn.SSConv
# conv_func = sscnn.e2cnn.PlainConv
backbone = C8Backbone(out_channels=2, conv_func=conv_func)
head = RegressionHead(backbone.out_type, conv_func)
model = torch.nn.Sequential(backbone, head)
model = model.to(device)

model.load_state_dict(torch.load('/mnt/workspace/sscnn/orient_state.pth'))

height, width = 33, 33
x = torch.rand(1, 1, height, width)

angle = 1.1
aff = torch.zeros(1, 2, 3)
aff[0, :, :2] = rot_mat(angle)
grid = F.affine_grid(aff, (1, 1) + image.shape, False)

rotated = F.grid_sample(x, grid, align_corners=False,
        padding_mode='zeros', mode='bilinear') 

y0 = model(x)
y1 = model(rotated)

import pdb; pdb.set_trace()
