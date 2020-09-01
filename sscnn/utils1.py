import math
import torch
from torch.nn import functional as F

def rot_mat(angle):
    angle = torch.tensor(angle)
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.Tensor([[c, -s], [s, c]])

def psi(order, k, theta):
    if k == 0:
        return torch.tensor(1).reshape(1, 1)
    elif k == order / 2:
        return torch.tensor(-1).reshape(1, 1)
    else:
        return rot_mat(k * theta)

def block_diag(blocks):
    i, j = 0, 0
    blocks = list(blocks)
    H = sum(b.shape[0] for b in blocks)
    W = sum(b.shape[1] for b in blocks)
    diag = torch.zeros(H, W)
    for block in blocks:
        assert block.dim() == 2
        h, w = block.shape
        diag[i:i + h, j:j + w] = block
        i, j = i + h, j + w
    return diag

def rotate_trivials(x, elem, order):
    if order % 4 == 0 and elem % (order // 4) == 0:
        quarter = order // 4
        rotated = x.rot90(elem // quarter, dims=[2, 3])
    else:
        angle = elem * math.pi * 2 / order
        c, s = math.cos(angle), math.sin(angle)
        aff = torch.Tensor([[[c, -s, 0], [s, c, 0]]]).expand(x.shape[0], -1, -1)
        grid = F.affine_grid(aff, x.shape, False)
        rotated = F.grid_sample(x, grid, align_corners=False)
    return rotated

def rotate_regulars(x, elem, order):
    N, C, H, W = x.shape
    assert C % order == 0

    rotated = rotate_trivials(x, elem, order).reshape(N, C // order, order, H, W)
    rotated = rotated.roll(elem, dims=2)

    return rotated.flatten(1, 2)

def rotate_irreps(x, elem, order, irreps):
    angle = elem * math.pi * 2 / order
    repr_mat = block_diag(psi(order, k, angle) for k in irreps)
    assert x.shape[1] == repr_mat.shape[1]

    rotated = rotate_trivials(x, elem, order)
    rotated = torch.einsum('cd,ndhw->nchw', repr_mat, rotated)
    return rotated
    
    
    

