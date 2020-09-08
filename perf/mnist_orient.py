import torch

from e2cnn import gspaces
from e2cnn import nn

import e2cnn
import sscnn.e2cnn

import numpy as np

from mnist_rot_dataset import MnistRotDataset
from models import C8SteerableCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

conv_func = nn.R2Conv
conv_func = sscnn.e2cnn.SSConv
# conv_func = sscnn.e2cnn.PlainConv
model = C8SteerableCNN(out_channels=10, conv_func=conv_func).to(device)

mnist_train = MnistRotDataset(mode='train')
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64)

mnist_test = MnistRotDataset(mode='test')
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64)

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)

for epoch in range(31):
    model.train()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for i, (x, l, v) in enumerate(train_loader):
        
        optimizer.zero_grad()

        x = x.to(device)
        l = l.to(device)
        v = v.to(device)

        y = model(x)
        y = F.normalize(y, dim=1)

        loss = loss_function(y, v)

        loss.backward()

        optimizer.step()

    end.record()
    torch.cuda.synchronize()
    print('Epoch', start.elapsed_time(end))
    
    if epoch % 10 == 0:
        total = 0
        error = 0
        with torch.no_grad():
            model.eval()
            for i, (x, t) in enumerate(test_loader):

                x = x.to(device)
                l = l.to(device)
                v = v.to(device)

                y = model(x)
                y = F.normalize(y, dim=1)

                _, prediction = torch.max(y.data, 1)
                total += t.shape[0]
                error += (y - v).square()
        print(f"epoch {epoch} | test accuracy: {error/total}")
