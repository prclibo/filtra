
import math
import numpy as np

import torch
from torch.nn import functional as F

from sscnn.rotater import *

group = (2, 8)
npatches = 96 * 64
size = (5, 5)
patch = torch.from_numpy(np.load('test/patch.npy'))[14:19, 14:19]
patch = torch.arange(25).float().reshape(size)

in_patch = patch[None, ...].expand(npatches, -1, -1)

rotater0 = FilterRotater(group, size, False, 'bmm')
rotater1 = FilterRotater(group, size, False, 'grid_sample')

out_patch0 = rotater0.forward(in_patch)
out_patch1 = rotater1.forward(in_patch)

import pdb; pdb.set_trace()

