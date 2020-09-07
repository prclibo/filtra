from typing import Tuple
import torch
from torch import nn
import e2cnn
from e2cnn.nn.modules.equivariant_module import EquivariantModule

import sscnn.conv
import time

def convert_field_type(type_):
    if type_.representations[0].name.startswith('irrep'):
        assert all(_.name.startswith('irrep') for _ in type_.representations)
        converted = [(
            rep.attributes.pop('flip_frequency', 0),
            rep.attributes.pop('frequency', 0)
        ) for rep in type_.representations]
    elif type_.representations[0].name.startswith('regular'):
        assert all(_.name.startswith('regular') for _ in type_.representations)
        converted = len(type_.representations)
    else:
        raise ValueError
    return converted

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

        self.in_type = in_type
        self.out_type = out_type
        in_type = convert_field_type(in_type)
        out_type = convert_field_type(out_type)

        conv_type = TYPE_TO_CONV[(type(in_type), type(out_type))]
        self.conv = conv_type(group, in_type, out_type,
                kernel_size=kernel_size, padding=padding,
                stride=stride, dilation=dilation, groups=groups, bias=bias)
    def forward(self, x):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        x = self.conv.forward(x.tensor)
        
        end.record()
        torch.cuda.synchronize()
        # print('    ', start.elapsed_time(end))

        return e2cnn.nn.GeometricTensor(x, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
    
        b, c, hi, wi = input_shape
        
        ho = math.floor((hi + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
        wo = math.floor((wi + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)

        return b, self.out_type.size, ho, wo
        
