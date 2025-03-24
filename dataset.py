import os
import random
import numpy as np
import torch
import cv2
import timm
import pandas as pd
import torchvision.transforms as transform
from sklearn import model_selection
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import zoom

class CustomDataset(Dataset):

    def __init__(self, df, data_path, transform=None):
        super().__init__()

        self.img_id = df['image'].values
        self.label = df['level'].values
        self.path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.img_id[idx] + '.jpeg')
        assert os.path.exists(img_path), '{} img path is not exists...'.format(img_path)

        label = self.label[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label


class SingleimgDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_dir, csv_list, transform=None):
        self.imgs_dir = imgs_dir
        self.csv_list = csv_list
        self.transform = transform
    def __len__(self):
        return len(self.csv_list)

    def __getitem__(self, idx):
        class_id = self.csv_list.iloc[idx, 1].astype(np.int64)


        img_name = self.csv_list.iloc[idx, 0]

        img_name = img_name if '.jpg' in img_name else img_name + '.jpg'
        img_path = os.path.join(self.imgs_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            print("image error ", img_path, "is not exist!")
            raise ValueError("image error ", img_path, "is not exist!")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, class_id


class MultiviewImgDataset_no_lesion(torch.utils.data.Dataset):

    def __init__(self, imgs_dir, data_list, scale_aug=False, rot_aug=False, test_mode=True, num_views=4,
                 transform=None):
        self.imgs_dir = imgs_dir
        self.data_list = data_list
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        self.transform = transform


    def __len__(self):
        return len(self.data_list) // 4

    def __getitem__(self, idx):

        class_id = self.data_list.iloc[idx * 4, 1].astype(np.int64)

        # Use PIL instead
        imgs = []


        for view in range(self.num_views):
            img_name = '{}.jpg'.format(self.data_list.iloc[idx * 4 + view, 0])
            img_path = os.path.join(self.imgs_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                print("image error ", img_path, "is not exist!")
                raise ValueError("image error ", img_path, "is not exist!")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image)  # transform
            imgs.append(image)
        return torch.stack(imgs), class_id


class MultiviewImgDataset(torch.utils.data.Dataset):

    def __init__(self, imgs_dir, data_list, scale_aug=False, rot_aug=False, test_mode=True, num_views=4,
                 transform=None):
        self.imgs_dir = imgs_dir
        self.data_list = data_list
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        self.transform = transform


    def __len__(self):
        return len(self.data_list) // 4

    def __getitem__(self, idx):

        class_id = self.data_list.iloc[idx * 4, 1].astype(np.int64)

        # Use PIL instead
        imgs = []

        # classes = self.classes
        for view in range(self.num_views):
            img_name = '{}.jpg'.format(self.data_list.iloc[idx * 4 + view, 0])
            img_path = os.path.join(self.imgs_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                print("image error ", img_path, "is not exist!")
                raise ValueError("image error ", img_path, "is not exist!")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image)  # transform
            imgs.append(image)

        return torch.stack(imgs), class_id


class MultiviewImgDataset_mask(torch.utils.data.Dataset):

    def __init__(self, imgs_dir, masks_dir, data_list, scale_aug=False, rot_aug=False, test_mode=True, num_views=4,
                 transform=None, Single=False,no_mask=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.data_list = data_list
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        self.transform = transform
        self.Single = Single
        self.no_mask=no_mask
    def __len__(self):
        return len(self.data_list) // 4

    def __getitem__(self, idx):

        class_id = self.data_list.iloc[idx * 4, 1].astype(np.int64)

        imgs = []
        for view in range(self.num_views):
            if self.Single:
                k = np.random.randint(0, 4)
                img_name = '{}.jpg'.format(self.data_list.iloc[idx * 4 + k, 0])
                mask_name = '{}.png'.format(self.data_list.iloc[idx * 4 + k, 0] + '_mask')
                img_path = os.path.join(self.imgs_dir, img_name)
                mask_path = os.path.join(self.masks_dir, mask_name)
            else:
                img_name = '{}.jpg'.format(self.data_list.iloc[idx * 4 + view, 0])
                mask_name = '{}.png'.format(self.data_list.iloc[idx * 4 + view, 0] + '_mask')
                img_path = os.path.join(self.imgs_dir, img_name)
                mask_path = os.path.join(self.masks_dir, mask_name)
            image = cv2.imread(img_path)
            if image is None:
                print("image error ", img_path, "is not exist!")
                raise ValueError("image error ", img_path, "is not exist!")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (1280, 1280))
            mask = cv2.imread(mask_path, 0)
            mask = mask.reshape((1280, 1280, 1))
            if self.no_mask:
                mask[:]=255
            hun = np.append(image, mask, axis=2)
            if self.transform:
                image = self.transform(hun)
            imgs.append(image)
        return torch.stack(imgs), class_id


class MultiviewImgDataset_noAggregate(torch.utils.data.Dataset):

    def __init__(self, imgs_dir, mask_dir, data_list, scale_aug=False, rot_aug=False, test_mode=True, num_views=4,
                 transform=None):
        self.imgs_dir = imgs_dir
        self.mask_dir = mask_dir
        self.data_list = data_list
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        self.transform = transform


    def __len__(self):
        return len(self.data_list) // 4

    def __getitem__(self, idx):

        class_id = self.data_list.iloc[idx * 4, 1].astype(np.int64)

        # Use PIL instead
        imgs = []

        for view in range(self.num_views):
            mask_name = '{}.png'.format(self.data_list.iloc[idx * 4 + view, 0] + '_mask')
            img_name = '{}.png'.format(self.data_list.iloc[idx * 4 + view, 0] + '_mask')
            img_path = os.path.join(self.imgs_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                print("image error ", img_path, "is not exist!")
                raise ValueError("image error ", img_path, "is not exist!")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image)  # transform
            imgs.append(image)


        return torch.stack(imgs), np.ones(4, dtype=np.int64) * class_id






