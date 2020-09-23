import torch

from e2cnn import gspaces
from e2cnn import nn

import e2cnn
import sscnn.e2cnn

class C8Backbone3x3(nn.SequentialModule):
    def __init__(self, out_channels, conv_func):
        super(C8Backbone3x3, self).__init__()

        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = gspaces.Rot2dOnR2(N=8)

        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 16 feature fields, each transforming under the regular representation of C8
        out_type = nn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        self.add_module('block1', nn.SequentialModule(
            # nn.MaskModule(in_type, 29, margin=1),
            conv_func(in_type, out_type, kernel_size=3, padding=1, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        ))
        in_type = out_type
        out_type = nn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        self.add_module('block1a', nn.SequentialModule(
            # nn.MaskModule(in_type, 29, margin=1),
            conv_func(in_type, out_type, kernel_size=3, padding=1, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        ))

        # convolution 2
        # the old output type is the input type to the next layer
        in_type = out_type
        # the output type of the second convolution layer are 32 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.add_module('block2', nn.SequentialModule(
            conv_func(in_type, out_type, kernel_size=3,  padding=2, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        ))
        self.add_module('pool1', nn.SequentialModule(
            nn.PointwiseMaxPool(out_type, kernel_size=3, stride=2)
        ))

        # convolution 3
        # the old output type is the input type to the next layer
        in_type = out_type
        # the output type of the third convolution layer are 32 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.add_module('block3', nn.SequentialModule(
            conv_func(in_type, out_type, kernel_size=3, padding=2, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        ))
        in_type = out_type
        # the output type of the third convolution layer are 32 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.add_module('block3a', nn.SequentialModule(
            conv_func(in_type, out_type, kernel_size=3, padding=2, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        ))


        # convolution 4
        # the old output type is the input type to the next layer
        in_type = out_type
        # the output type of the fourth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.add_module('block4', nn.SequentialModule(
            conv_func(in_type, out_type, kernel_size=3, padding=2, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        ))
        self.add_module('pool2', nn.SequentialModule(
            nn.PointwiseMaxPool(out_type, kernel_size=3, stride=2)
        ))

        # convolution 5
        # the old output type is the input type to the next layer
        in_type = out_type
        # the output type of the fifth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.add_module('block5', nn.SequentialModule(
            conv_func(in_type, out_type, kernel_size=3, stride=2, padding=2, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        ))
        # convolution 5
        # the old output type is the input type to the next layer
        in_type = out_type
        # the output type of the fifth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.add_module('block5a', nn.SequentialModule(
            conv_func(in_type, out_type, kernel_size=3, stride=2, padding=2, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        ))

        # convolution 6
        # the old output type is the input type to the next layer
        in_type = out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.add_module('block6', nn.SequentialModule(
            conv_func(in_type, out_type, kernel_size=3, padding=1, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        ))
        self.add_module('pool3', nn.PointwiseMaxPool(out_type, kernel_size=3, stride=1, padding=0))

        self.out_type = out_type

class Backbone5x5(nn.SequentialModule):
    def __init__(self, out_channels, conv_func, group):
        super(Backbone5x5, self).__init__()

        # the model is equivariant under rotations by 45 degrees, modelled by C8
        # self.r2_act = gspaces.Rot2dOnR2(N=8)
        self.r2_act = group

        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 16 feature fields, each transforming under the regular representation of C8
        out_type = nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])
        self.add_module('block1', nn.SequentialModule(
            # nn.MaskModule(in_type, 29, margin=1),
            conv_func(in_type, out_type, kernel_size=5, padding=1, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        ))

        # convolution 2
        # the old output type is the input type to the next layer
        in_type = out_type
        # the output type of the second convolution layer are 32 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        self.add_module('block2', nn.SequentialModule(
            conv_func(in_type, out_type, kernel_size=5, padding=2, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        ))
        self.add_module('pool1', nn.SequentialModule(
            nn.PointwiseMaxPool(out_type, kernel_size=3, stride=2)
        ))

        # convolution 3
        # the old output type is the input type to the next layer
        in_type = out_type
        # the output type of the third convolution layer are 32 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])
        self.add_module('block3', nn.SequentialModule(
            conv_func(in_type, out_type, kernel_size=5, padding=2, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        ))

        # convolution 4
        # the old output type is the input type to the next layer
        in_type = out_type
        # the output type of the fourth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.add_module('block4', nn.SequentialModule(
            conv_func(in_type, out_type, kernel_size=5, padding=2, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        ))
        self.add_module('pool2', nn.SequentialModule(
            nn.PointwiseMaxPool(out_type, kernel_size=3, stride=2)
        ))

        # convolution 5
        # the old output type is the input type to the next layer
        in_type = out_type
        # the output type of the fifth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.add_module('block5', nn.SequentialModule(
            conv_func(in_type, out_type, kernel_size=5, padding=2, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        ))

        # convolution 6
        # the old output type is the input type to the next layer
        in_type = out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.add_module('block6', nn.SequentialModule(
            conv_func(in_type, out_type, kernel_size=5, padding=1, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        ))
        self.add_module('pool3', nn.PointwiseMaxPool(out_type, kernel_size=3, stride=1, padding=0))

        self.out_type = out_type
        # self.gpool = nn.GroupPooling(out_type)
        # self.gpool = nn.PointwiseAdaptiveMaxPool(out_type, (1, 1))

        # # number of output channels
        # c = self.gpool.out_type.size

        # # Fully Connected
        # self.fully_net = torch.nn.Sequential(
        #     torch.nn.Linear(c, 64),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.ELU(inplace=True),
        #     torch.nn.Linear(64, out_channels),
        # )

class ClassificationHead(torch.nn.Module):
    pass

class RegressionHead(nn.SequentialModule):
    def __init__(self, in_type, conv_func):
        super(RegressionHead, self).__init__()
        gspace = in_type.gspace
        self.add_module('gpool', nn.PointwiseAdaptiveMaxPool(in_type, (1, 1)))
        # self.gpool = nn.GroupPooling(in_type)

        # number of output channels
        # Fully Connected
        in_type = in_type
        out_type = nn.FieldType(gspace, 32 * [gspace.regular_repr])
        self.add_module('block1', nn.SequentialModule(
            conv_func(in_type, out_type, kernel_size=1, padding=0, bias=False),
            nn.ReLU(out_type, inplace=True)
        ))
        in_type = out_type
        if isinstance(gspace, gspaces.Rot2dOnR2):
            out_type = nn.FieldType(gspace, [gspace.irrep(1)])
        elif isinstance(gspace, gspaces.FlipRot2dOnR2):
            out_type = nn.FieldType(gspace, [gspace.irrep(1, 1)])
        else:
            raise NotImplementedError

        self.add_module('block2', conv_func(in_type, out_type, kernel_size=1, padding=0, bias=False))

