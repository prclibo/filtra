import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from sscnn.utils import *

DATA_FOLDER='/data/'

def batch_rotate(images, angles):
    N, C, H, W = images.shape
    aff = images.new_zeros(N, 2, 3)
    cosa, sina = angles.cos(), angles.sin()
    aff[:, 0, 0] = cosa
    aff[:, 0, 1] = -sina
    aff[:, 1, 0] = sina
    aff[:, 1, 1] = cosa
    grid = F.affine_grid(aff, images.shape, False)
    rotated = F.grid_sample(images, grid, align_corners=False, 
            padding_mode='zeros', mode='bilinear') 
    return rotated

class RotatedDataset(Dataset):
    def __init__(self, dataset, random_rotate=True):
        super(RotatedDataset, self).__init__()
        self.dataset = dataset
        self.random_rotate = random_rotate

    def _rotate_data(self, angle, image):
        aff = torch.zeros(1, 2, 3)
        aff[0, :, :2] = rot_mat(angle)
        grid = F.affine_grid(aff, (1, 1) + image.shape, False)

        rotated = F.grid_sample(image[None, None, ...], grid, align_corners=False,
                padding_mode='zeros', mode='bilinear') 

        return rotated[0]

    def __getitem__(self, index):
        image, label = self.dataset[index].float(), int(self.targets[index])

        if self.random_rotate:
            angle = np.random.uniform(2 * np.pi)
        else:
            angle = 0.0
        # image = torch.from_numpy(image)# .transpose(0, 1)
        rotated = self._rotate_data(angle, image)
        orient = torch.Tensor([np.cos(angle), np.sin(angle)])

        # import matplotlib.pyplot as plt
        # plt.imshow(torch.cat([image, rotated[0]]).cpu())
        # plt.show()
        # print(orient)

        # import pdb; pdb.set_trace()

        return rotated, label, orient

class RotatedMNISTDataset(torchvision.datasets.MNIST):
    def __init__(self, root, train, random_rotate=True):
        super(RotatedMNISTDataset, self).__init__(root, train, download=True)
        print(self.data.shape, self.targets.shape)
        self.random_rotate = random_rotate

    def _rotate_data(self, angle, image):
        aff = torch.zeros(1, 2, 3)
        aff[0, :, :2] = rot_mat(angle)
        grid = F.affine_grid(aff, (1, 1) + image.shape, False)

        rotated = F.grid_sample(image[None, None, ...], grid, align_corners=False,
                padding_mode='zeros', mode='bilinear') 

        return rotated[0]

    def __getitem__(self, index):
        image, label = self.data[index].float(), int(self.targets[index])

        if self.random_rotate:
            angle = np.random.uniform(2 * np.pi)
        else:
            angle = 0.0
        # image = torch.from_numpy(image)# .transpose(0, 1)
        rotated = self._rotate_data(angle, image)
        orient = torch.Tensor([np.cos(angle), np.sin(angle)])

        # import matplotlib.pyplot as plt
        # plt.imshow(torch.cat([image, rotated[0]]).cpu())
        # plt.show()
        # print(orient)

        # import pdb; pdb.set_trace()

        return rotated, label, orient


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
        print(self.images.shape)

    def _rotate_data(self, angle, image):
        aff = torch.zeros(1, 2, 3)
        aff[0, :, :2] = rot_mat(angle)
        grid = F.affine_grid(aff, (1, 1) + image.shape, False)

        rotated = F.grid_sample(image[None, None, ...], grid, align_corners=False,
                padding_mode='zeros', mode='bilinear') 

        return rotated[0]

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]

        angle = np.random.uniform(2 * np.pi)
        image = torch.from_numpy(image).transpose(0, 1)
        rotated = self._rotate_data(angle, image)
        orient = torch.Tensor([np.cos(angle), np.sin(angle)])
        import matplotlib.pyplot as plt
        plt.imshow(torch.cat([image, rotated[0]]).cpu())
        plt.show()
        print(orient)

        import pdb; pdb.set_trace()

        return rotated , label, orient

    def __len__(self):
        return len(self.labels)


