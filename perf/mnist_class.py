import numpy as np
import torch

from e2cnn import gspaces
from e2cnn import nn

import e2cnn
import sscnn.e2cnn

from mnist_rot_dataset import MnistRotDataset, RotatedMNISTDataset, batch_rotate
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from models import C8SteerableCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128

conv_func = nn.R2Conv
conv_func = sscnn.e2cnn.SSConv
# conv_func = sscnn.e2cnn.PlainConv
model = C8SteerableCNN(out_channels=10, conv_func=conv_func).to(device)

# mnist_train = MnistRotDataset(mode='train')
# mnist_train = RotatedMNISTDataset('.', train=True)
mnist_train = MNIST('.', train=True, download=True, transform=ToTensor())
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size)

# mnist_test = MnistRotDataset(mode='test')
# mnist_test = RotatedMNISTDataset('.', train=False)
mnist_test = MNIST('.', train=False, download=True, transform=ToTensor())
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)

for epoch in range(31):
    model.train()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for i, (x, t) in enumerate(train_loader):
        
        optimizer.zero_grad()

        x = x.to(device)
        angles = torch.rand(len(x), device=device) * np.pi * 2
        x = batch_rotate(x, angles)
        t = t.to(device)

        y = model(x)
        # import pdb; pdb.set_trace()

        loss = loss_function(y, t)

        loss.backward()

        optimizer.step()
    end.record()
    torch.cuda.synchronize()
    print('Epoch', start.elapsed_time(end))
    
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
