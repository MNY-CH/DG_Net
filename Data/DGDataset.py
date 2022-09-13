import torch
from torch.utils.data import Dataset

import glob
import random
from PIL import Image

class DG_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.file_list = glob.glob(data_dir + '*')

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        while True:
            index2 = random.randrange(0, self.__len__())
            if index is not index2:
                break
        image_list_1 = glob.glob(self.file_list[index] + '/*.jpg')
        image_list_2 = glob.glob(self.file_list[index2] + '/*.jpg')

        while True:
            xi_index = random.randrange(0, len(image_list_2))
            xt_index = random.randrange(0, len(image_list_2))
            if xi_index is not xt_index:
                break
        xj = Image.open(image_list_1[random.randrange(0, len(image_list_1))])
        xi = Image.open(image_list_2[xt_index])
        xt = Image.open(image_list_2[xt_index])

        if self.transform:
            xj = self.transform(xj)
            xi = self.transform(xi)
            xt = self.transform(xt)
        xj_gt = index
        xi_gt =index2
        return xj_gt, xi_gt, xj, xi, xt
