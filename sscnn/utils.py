from typing import Tuple, Iterable, List, Union
import math
import torch
from torch.nn import functional as F

def check_validity(elem: Union[None, Tuple[int, int]],
        irrep: Union[None, Tuple[int, int]],
        group: Union[None, Tuple[int, int]]):
    assert group[0] in [1, 2] and group[1] >= 1
    assert not elem or all(e < g for (e, g) in zip(elem, group)),\
            f'irrep = {irrep}, group = {group}'
    # irrep[1] in range of [0, group[1] // 2]
    assert not irrep or all(e <= g // 2 and e >= 0 for (e, g) in zip(irrep, group)),\
            f'irrep = {irrep}, group = {group}'

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

def rot_mat(angle):
    angle = torch.tensor(angle)
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.Tensor([[c, -s], [s, c]])

def ref_mat(s):
    # return torch.Tensor([[(-1.)**s, 0.], [0., (-1.)**s]])
    return torch.Tensor([[1., 0.], [0., (-1.)**s]])

def tform_mat(elem: Tuple[int, int], irrep: Tuple[int, int], group: Tuple[int, int]):
    check_validity(elem, irrep, group)
    delta = math.pi * 2 / group[1]
    return rot_mat(irrep[1] * elem[1] * delta) @ ref_mat(irrep[0] * elem[0])

def irrep_mat(elem: Tuple[int, int], irrep: Tuple[int, int], group: Tuple[int, int]):
    check_validity(elem, irrep, group)
    # if irrep[1] in [0, group[1] / 2]:
    #     irrep_mat_s = torch.tensor(((-1) ** irrep[0]) ** elem[0])
    #     irrep_mat_r = torch.tensor(((-1)) ** irrep[1])
    #     return (irrep_mat_s * irrep_mat_r).reshape(1, 1)
    if irrep[1] == 0:
        return torch.tensor((((-1) ** irrep[0]) ** elem[0])).reshape(1, 1)
    elif irrep[1] == group[1] / 2:
        return torch.tensor(((-1) ** elem[1]) * (((-1) ** irrep[0]) ** elem[0])).reshape(1, 1)
    else:
        delta = math.pi * 2 / group[1]
        irrep_mat_r = rot_mat(irrep[1] * elem[1] * delta)
        irrep_mat_s = torch.eye(2) * ((-1)**irrep[0])
        # XXX Why???
        irrep_mat_s = ref_mat(elem[0]) * ((-1)**irrep[0])**elem[0]
        # irrep_mat_s = torch.eye(2) * ((-1)**irrep[0])
        # irrep_mat_s = ref_mat(irrep[0] * elem[0])
        return irrep_mat_r @ irrep_mat_s

def regular_mat(elem: Tuple[int, int], group: Tuple[int, int]):
    order = group[0] * group[1]
    identity = torch.eye(order)[..., None, None]
    rotated = rotate_regulars(identity, elem, group)
    return rotated[:, :, 0, 0].transpose(0, 1)

def rotate_trivials(x, elem: Tuple[int, int], group: Tuple[int, int]):
    aff = torch.zeros(x.shape[0], 2, 3)
    irrep = tuple(min(g - 1, 1) for g in group)
    aff[:, :, :2] = tform_mat(elem, irrep, group)
    grid = F.affine_grid(aff, x.shape, False)
    return F.grid_sample(x, grid, align_corners=False)

def rotate_regulars(x, elem: Tuple[int, int], group: Tuple[int, int]):
    N, C, H, W = x.shape
    order = group[0] * group[1]
    assert C % order == 0

    rotated = rotate_trivials(x, elem, group)
    rotated = rotated.reshape(N, C // order, group[0], group[1], H, W)
    if elem[0] == 0:
        rotated = rotated.roll(elem[1], dims=3).flatten(2, 3)
    else:
        rotated = rotated.roll(-((elem[1] + 1) % group[1]), dims=3).flatten(2, 3)
        rotated = rotated.flip(2)
    return rotated.flatten(1, 2)

def rotate_irreps(x, elem: Tuple[int, int], irreps: List[Tuple[int, int]], group: Tuple[int, int]):
    angle = math.pi * 2 * elem[1] / group[1]
    repr_mat = block_diag(irrep_mat(elem, irrep, group) for irrep in irreps)
    # print(repr_mat)
    assert x.shape[1] == repr_mat.shape[1]

    rotated = rotate_trivials(x, elem, group)
    rotated = torch.einsum('cd,ndhw->nchw', repr_mat, rotated)
    return rotated
    
    
