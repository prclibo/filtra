import torch

from e2cnn import gspaces
from e2cnn import nn

import e2cnn
import sscnn.e2cnn

class C8Backbone(torch.nn.Module):

    def __init__(self, out_channels, conv_func):

        super(C8Backbone, self).__init__()

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
        self.block1 = nn.SequentialModule(
            # nn.MaskModule(in_type, 29, margin=1),
            conv_func(in_type, out_type, kernel_size=5, padding=1, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 32 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            conv_func(in_type, out_type, kernel_size=5, padding=2, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 32 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block3 = nn.SequentialModule(
            conv_func(in_type, out_type, kernel_size=5, padding=2, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block4 = nn.SequentialModule(
            conv_func(in_type, out_type, kernel_size=5, padding=2, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block5 = nn.SequentialModule(
            conv_func(in_type, out_type, kernel_size=5, padding=2, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block6 = nn.SequentialModule(
            conv_func(in_type, out_type, kernel_size=5, padding=1, bias=False),
            # nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

        self.out_type = self.pool3.out_type
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

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        x = nn.GeometricTensor(input, self.input_type)

        # apply each equivariant block

        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)

        x = self.block5(x)
        x = self.block6(x)

        # pool over the spatial dimensions
        x = self.pool3(x)

        end.record()
        torch.cuda.synchronize()
        # print('-----', start.elapsed_time(end))

        return x

class ClassificationHead(torch.nn.Module):
    pass

class RegressionHead(torch.nn.Module):
    def __init__(self, in_type, conv_func):
        super(RegressionHead, self).__init__()
        gspace = in_type.gspace
        self.gpool = nn.PointwiseAdaptiveMaxPool(in_type, (1, 1))

        # number of output channels
        # Fully Connected
        in_type = self.gpool.out_type
        out_type = nn.FieldType(gspace, 8 * [gspace.regular_repr])
        self.block1 = nn.SequentialModule(
            conv_func(in_type, out_type, kernel_size=1, padding=0, bias=False),
            nn.ReLU(out_type, inplace=True)
        )
        in_type = self.block1.out_type
        out_type = nn.FieldType(gspace, [gspace.irrep(1)])
        self.block2 = conv_func(in_type, out_type, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x.tensor.flatten(1, -1)
