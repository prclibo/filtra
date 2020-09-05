import math
import numpy as np

import torch
from torch.nn import functional as F

order = 8
npatches = 96 * 64
patch = torch.from_numpy(np.load('test/patch.npy'))[14:19, 14:19].cuda()
# patch = torch.from_numpy(np.load('test/patch.npy')).cuda()

size = torch.tensor((2 * order, 1) + patch.shape)

aff = torch.zeros([2, order, 2, 3]) # reflection x order x 2 x 3
angles = torch.arange(order) * math.pi * 2 / order 
cos_na, sin_na = angles.cos(), angles.sin()
aff[0, :, 0, 0] = cos_na
aff[0, :, 0, 1] = -sin_na
aff[0, :, 1, 0] = sin_na
aff[0, :, 1, 1] = cos_na
aff[1, :, 0, 0] = cos_na
aff[1, :, 0, 1] = sin_na
aff[1, :, 1, 0] = sin_na
aff[1, :, 1, 1] = -cos_na

# [group[0] x group[1]] x H x W x 2
grid = F.affine_grid(aff.flatten(0, 1), size.tolist(), False).cuda()

# [group[0] x group[1]] x C (=1) x H x W
in_patch = patch[None, None, ...].expand(2 * order, npatches, -1, -1).cuda()

# [group[0] x group[1]] x C (=1) x H x W
for _ in range(10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out_patch = F.grid_sample(in_patch, grid, align_corners=False, padding_mode='zeros', mode='nearest') # mode='bilinear')

    end.record()
    torch.cuda.synchronize()
    print('    ', start.elapsed_time(end))

print('''-------------------------------''')

H, W = patch.shape
# import pdb ;pdb.set_trace()
grid = grid.mul(torch.Tensor([W / 2, H / 2])[None, None, None].cuda())
grid += torch.Tensor([(W - 1) / 2, (H - 1) / 2]).cuda()
grid = grid.round().long()

# H x W
disk_mask = (grid[:, :, :, 0] >= 0).all(0) &\
            (grid[:, :, :, 0] < H).all(0) &\
            (grid[:, :, :, 1] >= 0).all(0) &\
            (grid[:, :, :, 1] < W).all(0)
# [group[0] x group[1]] x H x W
grid = (grid[:, :, :, 1] * W + grid[:, :, :, 0])

# [group[0] x group[1]] x num_inbound x 1
grid = grid[:, disk_mask].unsqueeze(-1)
num_inbound = disk_mask.sum()

interp_mat_ = torch.zeros(2 * order, num_inbound, H * W).cuda()
interp_mat_.scatter_(2, grid, torch.ones_like(grid).float())

# [group[0] x group[1]] x [H x W] x [H x W]
interp_mat = torch.zeros(2 * order, H * W, H * W).cuda()
interp_mat[:, disk_mask.flatten(), :] = interp_mat_


# [group[0] x group[1]] x [H x W] x npatches
in_patch = patch.flatten()[None, ..., None].expand(2 * order, -1, npatches)
# # 1 x [group[0] x group[1] x H x W] x npatches
# in_patch = in_patch.flatten(0, 1).unsqueeze(0)
# # [group[0] x group[1] x H x W] x [H x W] x 1
# interp_mat = interp_mat.flatten(0, 1).unsqueeze(-1)

for _ in range(10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    out_patch = torch.bmm(interp_mat, in_patch)
    # out_patch = F.conv1d(in_patch, interp_mat, groups=2 * order)

    end.record()
    torch.cuda.synchronize()
    print('    ', start.elapsed_time(end))


print('''---------------------------------------''')

sparse_w = [m.to_sparse().coalesce() for m in interp_mat]

for _ in range(10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    # out_patch = torch.bmm(interp_mat, in_patch)

    out_patch = in_patch.new_zeros(2 * order, H * W, npatches)
    for n in range(2 * order):
        out_patch[n] = torch.sparse.mm(sparse_w[n], in_patch[n])

    # out_patch = F.conv1d(in_patch, interp_mat, groups=2 * order)

    end.record()
    torch.cuda.synchronize()
    print('    ', start.elapsed_time(end))


# import matplotlib.pyplot as plt
# # plt.imshow(disk_mask.detach())
# plt.imshow(out_patch[3].reshape(H, W))
# # plt.imshow(out_patch[1, 0].detach())
# plt.show()
import pdb; pdb.set_trace()


disk_mask = (grid.norm(dim=-1) > 1).unsqueeze(-1)
grid.masked_fill_(disk_mask, 100)
