
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

in_patch = patch[None, None, ...].expand(group[0] * group[1], npatches, -1, -1)

rotater = FilterRotater(group, size)

out_patch = rotater.forward(in_patch)
import pdb; pdb.set_trace()

