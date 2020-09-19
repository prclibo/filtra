import sys
sys.path.append('/mnt/workspace/sscnn/')

from collections import OrderedDict
import torch
import torch.nn.functional as F

from e2cnn import gspaces
from e2cnn import nn

import e2cnn
import sscnn.e2cnn

import numpy as np

from mnist_rot_dataset import RotatedMNISTDataset 
from models import C8Backbone, ClassificationHead, RegressionHead

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128

conv_func = nn.R2Conv
conv_func = sscnn.e2cnn.SSConv
# conv_func = sscnn.e2cnn.PlainConv
backbone = C8Backbone(out_channels=2, conv_func=conv_func)
head = RegressionHead(backbone.out_type, conv_func)
model = nn.SequentialModule(OrderedDict([
    ('backbone', backbone), ('head', head)
]))
model = model.to(device)

mnist_train = RotatedMNISTDataset('.', train=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size)

mnist_test = RotatedMNISTDataset('.', train=False)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size)

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)

for epoch in range(41):
    model.train()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for i, (x, l, v) in enumerate(train_loader):
        
        optimizer.zero_grad()

        x = x.to(device)
        v = v.to(device)

        y = model(nn.GeometricTensor(x, backbone.input_type))
        y = y.tensor.flatten(1, -1)
        y = F.normalize(y, dim=1)

        loss = loss_function(y, v)

        loss.backward()

        optimizer.step()
        # break

    end.record()
    torch.cuda.synchronize()
    print('Epoch', start.elapsed_time(end))

    exported = model.export()
    torch.save(exported.state_dict(), f'./orient_state_{conv_func.__name__}.pth')
    
    if epoch % 5 == 0:
        total = 0
        error = 0
        with torch.no_grad():
            model.eval()
            for i, (x, l, v) in enumerate(test_loader):

                x = x.to(device)
                v = v.to(device)

                y = model(nn.GeometricTensor(x, backbone.input_type))
                y = y.tensor.flatten(1, -1)
                y = F.normalize(y, dim=1)

                total += l.shape[0]

                angles = y.mul(v).sum(dim=1).clamp(-1, 1).acos() * 180 / np.pi
                error = angles.sum() + error

                loss = loss_function(y, v)
        print(f"epoch {epoch} | tes : {error/total}, {loss}")
