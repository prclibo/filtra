import sys
sys.path.append('/mnt/workspace/sscnn/')

import os
import time
import argparse

from collections import OrderedDict
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from e2cnn import gspaces
from e2cnn import nn

import e2cnn
import sscnn.e2cnn

import numpy as np

from mnist_rot_dataset import RotatedMNISTDataset, TransformedDataset
from models import C8Backbone3x3, Backbone5x5, ClassificationHead, RegressionHead

from functools import partial

import matplotlib.pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--reflection', type=int, default=1)
    parser.add_argument('--rotation', type=int)
    parser.add_argument('--rotate_data', action='store_true')
    parser.add_argument('--reflect_data', action='store_true')
    parser.add_argument('--conv', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--dataset', type=str)
    return parser

def abs_included_angles(y, l, v):
    y = F.normalize(y, dim=1)
    angles = y.mul(v).sum(dim=1).clamp(-1, 1).acos() * 180 / np.pi
    return angles.squeeze().tolist()

def mask_of_success(y, l, v):
    _, prediction = torch.max(y.data, 1)
    mask = prediction == l
    return mask.squeeze().tolist()

def train(args):
    torch.manual_seed(23)
    torch.cuda.manual_seed(23)
    np.random.seed(23)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    
    if args.reflection == 1:
        group=gspaces.Rot2dOnR2(N=args.rotation)
    elif args.reflection == 2:
        group=gspaces.FlipRot2dOnR2(N=args.rotation)
    else:
        raise NotImplementedError
    
    if args.conv == 'R2Conv':
        conv_func = nn.R2Conv
    elif args.conv == 'SSConv':
        conv_func = sscnn.e2cnn.SSConv
    elif args.conv == 'PlainConv':
        conv_func = sscnn.e2cnn.PlainConv
    else:
        raise NotImplementedError

    backbone = Backbone5x5(out_channels=2, conv_func=conv_func, group=group)
    if args.task == 'classification':
        head = ClassificationHead(backbone.out_type, num_classes=10)
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        loss_function = lambda y, l, v: cross_entropy_loss(y, l)
        eval_function = mask_of_success
    elif args.task == 'regression':
        head = RegressionHead(backbone.out_type, conv_func)
        mse_loss = torch.nn.MSELoss()
        loss_function = lambda y, l, v: mse_loss(y, v)
        eval_function = abs_included_angles
    else:
        raise NotImplementedError

    if args.dataset == 'MNIST':
        dataset_func = partial(torchvision.datasets.MNIST,
                transform=transforms.ToTensor())
    elif args.dataset == 'CIFAR10':
        dataset_func = partial(torchvision.datasets.CIFAR10,
                transform=transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
        )
    else:
        raise NotImplementedError
    
    model = nn.SequentialModule(OrderedDict([
        ('backbone', backbone), ('head', head)
    ]))
    model = model.to(device)
    
    mnist_train = TransformedDataset(
            dataset_func('.', train=True, download=True),
            random_rotate=args.rotate_data, random_reflect=args.reflect_data)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size)
    
    mnist_test = TransformedDataset(
            dataset_func('.', train=False, download=True),
            random_rotate=args.rotate_data, random_reflect=args.reflect_data)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    
    file_path = None
    for epoch in range(41):
        model.train()
    
        if device == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = time.time()
    
        for i, (x, l, v) in enumerate(train_loader):
            
            optimizer.zero_grad()
    
            x = x.to(device)
            l = l.to(device)
            v = v.to(device)
    
            y = model(nn.GeometricTensor(x, backbone.input_type))
            y = y.tensor.flatten(1, -1)
    
            loss = loss_function(y, l, v)
    
            loss.backward()
    
            optimizer.step()
            # if i > 50:
            #     break
    
        if device == 'cuda':
            end.record()
            torch.cuda.synchronize()
            print('Epoch', start.elapsed_time(end))
        else:
            end = time.time()
            print('Epoch', end - start)
    
        if epoch % 5 == 0:
            errors = []
            with torch.no_grad():
                model.eval()
                for i, (x, l, v) in enumerate(test_loader):
    
                    x = x.to(device)
                    l = l.to(device)
                    v = v.to(device)
    
                    y = model(nn.GeometricTensor(x, backbone.input_type))
                    y = y.tensor.flatten(1, -1)

                    res = eval_function(y, l, v)
                    errors.extend(res)

            error = np.array(errors).mean()
            print(f"epoch {epoch} | tes : {error}")
            exported = model.export()
            print(file_path)
            if file_path:
                print('removing')
                os.remove(file_path)
            file_path = f'./orient_state_{conv_func.__name__}.{type(group).__name__}.{error}.pth'
            torch.save(exported.state_dict(), file_path)
        

if __name__ == '__main__':
    args = get_parser().parse_args()
    print(args)

    train(args)
