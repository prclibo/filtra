import torch

from e2cnn import gspaces
from e2cnn import nn

import e2cnn
import sscnn.e2cnn

from mnist_rot_dataset import MnistRotDataset, RotatedMNISTDataset
from models import C8SteerableCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

conv_func = nn.R2Conv
conv_func = sscnn.e2cnn.SSConv
# conv_func = sscnn.e2cnn.PlainConv
model = C8SteerableCNN(out_channels=10, conv_func=conv_func).to(device)

mnist_train = MnistRotDataset(mode='train')
# mnist_train = RotatedMNISTDataset('.', train=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64)

mnist_test = MnistRotDataset(mode='test')
# mnist_test = RotatedMNISTDataset('.', train=False)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)

for epoch in range(31):
    model.train()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for i, (x, t, v) in enumerate(train_loader):
        
        optimizer.zero_grad()

        x = x.to(device)
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
            for i, (x, t, v) in enumerate(test_loader):

                x = x.to(device)
                t = t.to(device)
                
                y = model(x)

                _, prediction = torch.max(y.data, 1)
                total += t.shape[0]
                correct += (prediction == t).sum().item()
        print(f"epoch {epoch} | test accuracy: {correct/total*100.}")
