import torch.nn as tnn
import e2cnn.nn as enn

def is_regular(rep):
    return True

class SingleBlockFilterStacker(tnn.Module):
    def __init__(self, in_repr, out_repr, k_size):
        self.in_repr = in_repr
        self.out_repr = out_repr
        self.k_size = k_size

        self.in_is_regu = is_regular(in_repr)
        self.out_is_regu = is_regular(out_repr)
        assert self.in_is_regu or self.out_is_regu
        assert in_repr.group.order() == out_repr.group.order()

    def forward(self, weight):
        '''
        weight: (out_repr.size or in_repr.size) * H * W
        '''
        filter_ = weight.new_zeros(
                [self.out_repr.size, self.in_repr.size, self.k_size, self.k_size])
        if self.in_is_regu:
            for out_ind, out_irrep in enumerate(self.out_repr.irreps):
                kernel = weight[out_ind, :, :]
                for in_irrep in self.in_repr.irreps:
                    if irrep == 'irrep_0':
                        filter_[out_ind, :, :, :] = self.stacked_filter
                    else:
                        out_ind0 = None
                        out_ind1 = None
                        freq = None
                        filter_[out_ind0, :, :, :] = self.stacked_filter * cos_nx
                        filter_[out_ind1, :, :, :] = self.stacked_filter * sin_nx

        elif self.out_is_regu:
            for out_irrep in self.out_repr.irreps:
                kernel = weight[in_ind, :, :]
                for in_ind, in_irrep in enumerate(self.in_repr.irreps):
                    if irrep == 'irrep_0':
                        filter_[:, in_ind, :, :] = self.stacked_filter
                    else:
                        in_ind0 = None
                        in_ind1 = None
                        freq = None
                        filter_[:, in_ind0, :, :] = self.stacked_filter * cos_nx
                        filter_[:, in_ind1, :, :] = self.stacked_filter * sin_nx


class BlocksFilterStacker(tnn.Module):
    def __init__(self, in_type, out_type, k_size):
        self._in_type = in_type
        self._out_type = out_type
        self._in_starts = [_.size for _ in in_type]
        self._out_starts = [_.size for _ in out_type]
        

    def forward(self, weight):
        '''
        weight: out_type.size * in_type.size * H * W
        '''
        # Check that only one side can have non-regular repr

        # 

        for in_repr in in_type.representations:
            for out_repr in out_type.representations:
                in_ind = None
                out_ind = None

                stacker = FILTER_STACKERS.find(out_repr, in_repr, k_size)
                kernel = weight[]
                filter_[o0:o1, i0:i1, :, :] = stacker()

