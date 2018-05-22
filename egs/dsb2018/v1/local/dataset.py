# Copyright 2018 Johns Hopkins University (author: Yiwen Shao)
# Apache 2.0

""" This module provides a pytorch-fashion customized dataset class
"""

import torch
<<<<<<< HEAD
import numpy as np


class Dataset_dsb2018():
    def __init__(self, path, transformation, offset_list,
                 num_classes, height, width):
        # self.data is a dictionary with keys ['img', 'mask'] and probably
        # 'class_object' in the future
        self.data = torch.load(path)
        self.transformation = transformation
        self.offset_list = offset_list
        self.num_classes = num_classes
        self.height = height
        self.width = width

    def __getitem__(self, index):
        data = self.data[index]
        # input images
        img = data['img'].numpy()
        height, width, channel = img.shape
        img = self.transformation(img)

        # bounding box
        num_offsets = len(self.offset_list)
        mask = data['mask'].numpy()
        bound = torch.zeros(num_offsets, self.height, self.width)

        # getting offset feature maps (i.e. bound)
        for k in range(num_offsets):
            i, j = self.offset_list[k]
            # roll the mask in rows and columns according to the offset
            rolled_mask = np.roll(np.roll(mask, i, axis=1), j, axis=0)
            # compare mask and the rolled mask to get whether pixel (x,y) is
            # of the same object as pixel (x+i, y+j)
            bound_unscaled = (torch.FloatTensor(
                (rolled_mask == mask).astype('float'))).unsqueeze(0)
            # do same transformation on the bounds as on images
            bound[k:k + 1] = self.transformation(bound_unscaled)

        # class label
        class_label = torch.zeros((self.num_classes, self.height, self.width))
        for c in range(self.num_classes):
            if c == 0:
                class_label_unscaled = (torch.FloatTensor(
                    (mask == 0).astype('float'))).unsqueeze(0)
            else:  # TODO, the current version is for 2 classes only
                class_label_unscaled = (torch.FloatTensor(
                    (mask > 0).astype('float'))).unsqueeze(0)
            class_label[c:c +
                        1] = self.transformation(class_label_unscaled)
=======
from torch.utils.data import Dataset, DataLoader
from waldo.data_manipulation import convert_to_combined_image
from waldo.data_transformation import randomly_crop_combined_image


class Dataset_dsb2018(Dataset):
    def __init__(self, path, c_cfg, size):
        # self.data is a dictionary with keys ['id', 'img', 'mask', 'object_class']
        self.data = torch.load(path)
        self.c_cfg = c_cfg
        self.size = size

    def __getitem__(self, index):
        data = self.data[index]
        combined_img = convert_to_combined_image(data, self.c_cfg)
        n_classes = self.c_cfg.num_classes
        n_offsets = len(self.c_cfg.offsets)
        n_colors = self.c_cfg.num_colors
        cropped_img = randomly_crop_combined_image(
            combined_img, self.c_cfg, self.size, self.size)

        img = torch.from_numpy(
            cropped_img[:n_colors, :, :]).type(torch.FloatTensor)
        class_label = torch.from_numpy(
            cropped_img[n_colors:n_colors + n_classes, :, :]).type(torch.FloatTensor)
        bound = torch.from_numpy(
            cropped_img[n_colors + n_classes:n_colors +
                        n_classes + n_offsets, :, :]).type(torch.FloatTensor)
>>>>>>> waldo-seg/master

        return img, class_label, bound

    def __len__(self):
        return len(self.data)
<<<<<<< HEAD
=======


if __name__ == '__main__':
    from waldo.core_config import CoreConfig
    import torchvision
    c_config = CoreConfig()
    c_config.read('exp/unet_5_10_sgd/configs/core.config')
    trainset = Dataset_dsb2018('data/train_val/train.pth.tar',
                               c_config, 128)
    trainloader = DataLoader(
        trainset, num_workers=1, batch_size=16, shuffle=True)
    data_iter = iter(trainloader)
    # data_iter.next()
    img, class_label, bound = data_iter.next()
    # torchvision.utils.save_image(class_label[:, 0:1, :, :], 'class0.png')
    # torchvision.utils.save_image(class_label[:, 1:2, :, :], 'class1.png')
    # torchvision.utils.save_image(bound[:, 0:1, :, :], 'bound0.png')
    # torchvision.utils.save_image(bound[:, 1:2, :, :], 'bound1.png')
>>>>>>> waldo-seg/master
