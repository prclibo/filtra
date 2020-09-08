import numpy as np
import torch
from sscnn.utils import *

DATA_FOLDER='/data/'

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

    def _rotate_data(self, angle, image):
        aff = torch.zeros(1, 2, 3)
        aff[0, :, :2] = rot_mat(angle)
        grid = F.affine_grid(aff, (1, 1) + image.shape, False)

        image = F.grid_sample(image[None, None, ...], align_corners=False,
                padding_mode='zeros', mode='bilinear') 

        return image

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]

        angle = np.random.uniform(2 * np.pi)
        image = self._rotate_data(angle, image)
        orient = torch.Tensor([np.cos(angle), np.sin(angle)])

        return image, orient, label

    def __len__(self):
        return len(self.labels)


