import time

import torch
from torch import nn

import e2cnn
import sscnn.conv

from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose

import numpy as np

from PIL import Image

DATA_FOLDER = '/home/li/data/'
APPROACH = 'e2cnn'
# APPROACH = 'torch'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_e2cnn_type(group, type_):
    g = e2cnn.gspaces.Rot2dOnR2(N=group[1])
    if isinstance(type_, int):
        return e2cnn.nn.FieldType(g, type_ * [g.regular_repr])
    elif isinstance(type_, list):
        assert len(type_) == 1 and type_[0] == (0, 0)
        return e2cnn.nn.FieldType(g, [g.trivial_repr])
    else:
        raise NotImplementedError

def conv_block_e2cnn(group, in_type, out_type, kernel_size, padding, bias):
    assert group[0] == 1
    g = e2cnn.gspaces.Rot2dOnR2(N=group[1])
    types = []
    in_type = get_e2cnn_type(group, in_type)
    out_type = get_e2cnn_type(group, out_type)
    conv = e2cnn.nn.SequentialModule(
        e2cnn.nn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=padding, bias=bias),
        # e2cnn.nn.InnerBatchNorm(out_type),
        e2cnn.nn.ReLU(out_type, inplace=True)
    )
    return conv

def poolaa_block_e2cnn(group, out_type, sigma, stride, padding=None):
    out_type = get_e2cnn_type(group, out_type)
    pool = e2cnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=sigma, stride=stride, padding=padding)
    return pool

def pool_block_e2cnn(group, out_type, kernel_size, stride, padding=0):
    out_type = get_e2cnn_type(group, out_type)
    return e2cnn.nn.PointwiseAvgPool(out_type, kernel_size=kernel_size, stride=stride, padding=padding)

def gpool_block_e2cnn(group, out_type):
    out_type = get_e2cnn_type(group, out_type)
    return e2cnn.nn.GroupPooling(out_type)

def input_func_e2cnn(x, group, in_type):
    in_type = get_e2cnn_type(group, in_type)
    return e2cnn.nn.GeometricTensor(x, in_type)

def output_func_e2cnn(x):
    return x.tensor

class AsTorchModule(nn.Module):
    def __init__(self, in_type, e2cnn_module):
        super(AsTorchModule, self).__init__()

        self.in_type = in_type 
        self.e2cnn_module = e2cnn_module
    def forward(self, x):
        x = self.e2cnn_module.forward(e2cnn.nn.GeometricTensor(x, self.in_type))
        return x.tensor

def gpool_block_torch(group, in_type):
    in_type = get_e2cnn_type(group, in_type)
    m = AsTorchModule(in_type, e2cnn.nn.GroupPooling(in_type))
    return m

def input_func_torch(x, group, in_type):
    return x

def output_func_torch(x):
    return x

def conv_block_torch(group, in_type, out_type, kernel_size, padding, bias):
    assert group[0] == 1

    if isinstance(in_type, list) and isinstance(out_type, int):
        conv = sscnn.conv.IrrepToRegular
    elif isinstance(in_type, int) and isinstance(out_type, list):
        conv = sscnn.conv.RegularToIrrep
    elif isinstance(in_type, int) and isinstance(out_type, int):
        conv = sscnn.conv.RegularToRegular
    e2cnn_out_type = get_e2cnn_type(group, out_type)
    conv = nn.Sequential(
        conv(group, in_type, out_type, kernel_size=kernel_size, padding=padding, bias=bias),
        # AsTorchModule(e2cnn_out_type, e2cnn.nn.InnerBatchNorm(e2cnn_out_type)),
        nn.ReLU(inplace=True)
    )
    return conv

def poolaa_block_torch(group, in_type, sigma, stride, padding=None):
    in_type = get_e2cnn_type(group, in_type)
    m = AsTorchModule(
        in_type, e2cnn.nn.PointwiseAvgPoolAntialiased(
            in_type, sigma=sigma, stride=stride, padding=padding))
    return m


conv_block = locals()[f'conv_block_{APPROACH}']
poolaa_block = locals()[f'poolaa_block_{APPROACH}']
gpool_block = locals()[f'gpool_block_{APPROACH}']
input_func = locals()[f'input_func_{APPROACH}']
output_func = locals()[f'output_func_{APPROACH}']

class Net(nn.Module):
    def __init__(self, n_classes=10):

        super(Net, self).__init__()
        group = (1, 8)
        self.group = group
        self.in_type = [(0, 0)]
        self.block1 = conv_block(group, [(0, 0)], 24, kernel_size=7, padding=1, bias=False)
        self.block2 = conv_block(group, 24, 48, kernel_size=5, padding=2, bias=False)
        self.pool1 = poolaa_block(group, 48, sigma=0.66, stride=2)
        self.block3 = conv_block(group, 48, 48, kernel_size=5, padding=2, bias=False)
        self.block4 = conv_block(group, 48, 96, kernel_size=5, padding=2, bias=False)
        self.pool2 = poolaa_block(group, 96, sigma=0.66, stride=2)
        self.block5 = conv_block(group, 96, 96, kernel_size=5, padding=2, bias=False)
        self.block6 = conv_block(group, 96, 64, kernel_size=5, padding=1, bias=False)
        self.pool3 = poolaa_block(group, 64, sigma=0.66, stride=1, padding=0)
        self.gpool = gpool_block(group, 64)
        self.fully_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.Linear(64, n_classes),
        )

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        import time; start = time.time()
        x = input_func(input, self.group, self.in_type)

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

        # pool over the group
        x = self.gpool(x)
        x = output_func(x)

        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x

model = Net().to(device)

class MnistRotDataset(Dataset):

    def __init__(self, mode, transform=None):
        assert mode in ['train', 'test']

        if mode == "train":
            file = DATA_FOLDER + "mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
        else:
            file = DATA_FOLDER + "mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"

        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')

        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)

print('Loaded')

# images are padded to have shape 29x29.
# this allows to use odd-size filters with stride 2 when downsampling a feature map in the model
pad = Pad((0, 0, 1, 1), fill=0)

# to reduce interpolation artifacts (e.g. when testing the model on rotated images),
# we upsample an image by a factor of 3, rotate it and finally downsample it again
resize1 = Resize(87)
resize2 = Resize(29)

totensor = ToTensor()

train_transform = Compose([
    pad,
    resize1,
    RandomRotation(180, resample=Image.BILINEAR, expand=False),
    resize2,
    totensor,
])

mnist_train = MnistRotDataset(mode='train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64)


test_transform = Compose([
    pad,
    totensor,
])
mnist_test = MnistRotDataset(mode='test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)

for epoch in range(31):
    model.train()
    start_time = time.time()
    for i, (x, t) in enumerate(train_loader):
        
        optimizer.zero_grad()

        x = x.to(device)
        t = t.to(device)

        y = model(x)

        loss = loss_function(y, t)

        loss.backward()

        optimizer.step()

    end_time = time.time()
    print('Epoch elapse', end_time - start_time)
    
    if epoch % 10 == 0:
        total = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            for i, (x, t) in enumerate(test_loader):

                x = x.to(device)
                t = t.to(device)
                
                y = model(x)

                _, prediction = torch.max(y.data, 1)
                total += t.shape[0]
                correct += (prediction == t).sum().item()
        print(f"epoch {epoch} | test accuracy: {correct/total*100.}")
