from typing import Tuple
import torch
from torch import nn
import e2cnn
from e2cnn.nn.modules.equivariant_module import EquivariantModule

import sscnn.conv
import time

def convert_field_type(field):
    if field.representations[0].name.startswith('irrep'):
        assert all(_.name.startswith('irrep') for _ in field.representations)
        converted = [(
            rep.attributes.pop('flip_frequency', 0),
            rep.attributes.pop('frequency', 0)
        ) for rep in field.representations]
        mult = len(converted)
        dim = 2
    elif field.representations[0].name.startswith('regular'):
        assert all(_.name.startswith('regular') for _ in field.representations)
        converted = len(field.representations)
        mult = converted
        dim = field.representations[0].group.order()
    else:
        raise ValueError
    return converted, mult, dim

def type_to_selection(type_, group):
    if type(type_) == int:
        sel = torch.arange(type_ * group[0] * group[1])
        length = type_ * group[0] * group[1]
    elif type(type_) == list:
        sel = []
        for i, irrep in enumerate(type_):
            sel.append(2 * i)
            if not irrep[1] in [0, group[1] / 2]:
                sel.append(2 * i + 1)
        sel = torch.tensor(sel)
        length = len(type_) * 2
    else:
        raise ValueError
    return sel

TYPE_TO_CONV = {
    (int, list): sscnn.conv.RegularToIrrep,
    (list, int): sscnn.conv.IrrepToRegular,
    (int, int): sscnn.conv.RegularToRegular,
}

class SSConv(EquivariantModule):
    def __init__(self,
        in_type: e2cnn.nn.FieldType,
        out_type: e2cnn.nn.FieldType,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(SSConv, self).__init__()
        if isinstance(in_type.gspace, e2cnn.gspaces.Rot2dOnR2):
            assert isinstance(in_type.gspace.fibergroup, e2cnn.group.CyclicGroup)
            group = (1, len(in_type.gspace.fibergroup.elements))
        else:
            raise NotImplementedError

        self.group = group
        self.in_type = in_type
        self.out_type = out_type
        in_type, self.in_mult, self.in_inner_dim = convert_field_type(in_type)
        out_type, self.out_mult, self.out_inner_dim = convert_field_type(out_type)

        in_sel = type_to_selection(in_type, group)
        out_sel = type_to_selection(out_type, group)
        self.register_buffer('in_sel', in_sel)
        self.register_buffer('out_sel', out_sel)

        conv_type = TYPE_TO_CONV[(type(in_type), type(out_type))]
        self.conv = conv_type(group, in_type, out_type,
                kernel_size=kernel_size, padding=padding,
                stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.bias = None

    def forward(self, x):
        '''
        Args:
            x: N x [in_mult x group[0] x group[1]] x H x W or N x [in_dim] x H x W
        '''

        N, C, H, W = x.shape
        order = self.group[0] * self.group[1]

        x = x.tensor
        x_ = x.new_zeros(N, self.in_mult * self.in_inner_dim, H, W)
        x_.index_copy_(1, self.in_sel, x)
        x_ = x_.view(N, self.in_mult, self.in_inner_dim, H, W)
        x_ = x_.transpose(1, 2).reshape(N, -1, H, W)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        x_ = self.conv.forward(x_)
        
        end.record()
        torch.cuda.synchronize()
        # print('    ', start.elapsed_time(end))

        N, C, H, W = x_.shape
        x_ = x_.view(N, self.out_inner_dim, self.out_mult, H, W)
        x_ = x_.transpose(1, 2).reshape(N, -1, H, W)
        x = x_.index_select(1, self.out_sel)

        return e2cnn.nn.GeometricTensor(x, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        b, c, hi, wi = input_shape
        
        ho = math.floor((hi + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
        wo = math.floor((wi + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)

        return b, self.out_type.size, ho, wo

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.Conv2d` module and set to "eval" mode.

        """
        
        # set to eval mode so the filter and the bias are updated with the current
        # values of the weights
        return SSConvExported(self)

class SSConvExported(torch.nn.Module):
    def __init__(self, conv):
        super(SSConvExported, self).__init__()
        self.conv = conv
    def forward(self, x):
        return self.conv.forward(e2cnn.nn.GeometricTensor(x, self.conv.in_type)).tensor

        
class PlainConv(EquivariantModule):
    def __init__(self,
        in_type: e2cnn.nn.FieldType,
        out_type: e2cnn.nn.FieldType,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(PlainConv, self).__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.conv = torch.nn.Conv2d(in_type.size, out_type.size, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias)

    def forward(self, x):
        x = self.conv.forward(x.tensor)
        return e2cnn.nn.GeometricTensor(x, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        b, c, hi, wi = input_shape
        
        ho = math.floor((hi + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
        wo = math.floor((wi + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)

        return b, self.out_type.size, ho, wo
