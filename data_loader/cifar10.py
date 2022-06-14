import sys

import numpy as np
from PIL import Image
import torchvision
from torch.utils.data.dataset import Subset
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances 
import torch
import torch.nn.functional as F
import random
import json
import os

def get_cifar10(root, cfg_trainer, train=True,
                transform_train=None, transform_train_aug=None, transform_val=None,
                download=False, noise_file = '',validation_split=0):
    base_dataset = torchvision.datasets.CIFAR10(root, train=train, download=download)
    if train:
        # train_idxs, val_idxs = train_val_split(base_dataset.targets, validation_split=validation_split)
        # train_idxs = np.sort(train_idxs)
        val_idxs = []
        train_idxs = np.arange(len(base_dataset.targets))

        train_dataset = CIFAR10_train(root, cfg_trainer, train_idxs, train=True, transform=transform_train, transform_aug=transform_train_aug)
        val_dataset = CIFAR10_val(root, cfg_trainer, val_idxs, train=train, transform=transform_val)
        if cfg_trainer['asym']:
            train_dataset.asymmetric_noise()
            val_dataset.asymmetric_noise()
        elif cfg_trainer['instance']:
            train_dataset.instance_noise()
            val_dataset.instance_noise()
        elif cfg_trainer['real'] is not None:
            noise_type = cfg_trainer['real']
            new_targets = cifar10n(root, noise_type, download=download)
            train_dataset.train_labels = np.array(new_targets)[train_dataset.indexs]
            val_dataset.train_labels = np.array(new_targets)[val_dataset.indexs]
        else:
            train_dataset.symmetric_noise()
            val_dataset.symmetric_noise()

        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")  # Train: 45000 Val: 5000
        val_dataset.noise_or_not = np.transpose(val_dataset.train_labels)==np.transpose(val_dataset.train_labels_gt)
        train_dataset.noise_or_not = np.transpose(train_dataset.train_labels)==np.transpose(train_dataset.train_labels_gt)
    else:
        train_dataset = []
        val_dataset = CIFAR10_val(root, cfg_trainer, None, train=train, transform=transform_val)
        print(f"Test: {len(val_dataset)}")



    return train_dataset, val_dataset



def download_cifarn(root):
        # wget.download('http://128.114.59.66:5000/files/CIFAR-N.zip', out=root)
        wget.download('http://ucsc-real.soe.ucsc.edu:1995/files/cifar-10-100n-main.zip', out=root)
        with ZipFile(os.path.join(root, 'cifar-10-100n-main.zip'), 'r') as f:
            f.extractall(root)

def cifar10n(root, key, download=False):
    label_path = os.path.join(root, 'CIFAR-10_human.pt')
    if not os.path.exists(label_path):
        if download:
            download_cifarn(root)
        else:
            raise FileNotFoundError(f'Labels not found. Please set download=True. Path: {data_path}')
    noise_label = torch.load(label_path)
    targets_new = torch.from_numpy(noise_label[key])
    return targets_new

def train_val_split(base_dataset: torchvision.datasets.CIFAR10, validation_split=0):
    num_classes = 10
    base_dataset = np.array(base_dataset)
    # train_n = int(len(base_dataset) * 0.9 / num_classes)
    train_n = int(len(base_dataset) * (1.-validation_split) / num_classes)
    train_idxs = []
    val_idxs = []

    for i in range(num_classes):
        idxs = np.where(base_dataset == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs


class CIFAR10_train(torchvision.datasets.CIFAR10):
    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, transform_aug = None, target_transform=None,
                 download=False):
        super(CIFAR10_train, self).__init__(root, train=train,
                                            transform=transform,
                                            target_transform=target_transform,
                                            download=download)
        self.num_classes = 10
        self.cfg_trainer = cfg_trainer
        self.train_data = self.data[indexs]#self.train_data[indexs]
        self.train_labels = np.array(self.targets)[indexs]#np.array(self.train_labels)[indexs]
        self.indexs = indexs
        self.prediction = np.zeros((len(self.train_data), self.num_classes, self.num_classes), dtype=np.float32)
        self.noise_indx = []
        self.transform_aug = transform_aug

        self.train_labels_gt = self.train_labels.copy()
        self.noise_or_not = None
        
    def symmetric_noise(self):
        # self.train_labels_gt = self.train_labels.copy()
        #np.random.seed(seed=888)
        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.cfg_trainer['percent'] * len(self.train_data):
                self.noise_indx.append(idx)
                self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)

    def asymmetric_noise(self):
        # self.train_labels_gt = self.train_labels.copy()
        for i in range(self.num_classes):
            indices = np.where(self.train_labels == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.cfg_trainer['percent'] * len(indices):
                    self.noise_indx.append(idx)
                    # truck -> automobile
                    if i == 9:
                        self.train_labels[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.train_labels[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.train_labels[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.train_labels[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.train_labels[idx] = 7

    def instance_noise(self):


        noise_label = torch.load(self.root + 'cifar-noisy/IDN_{:.1f}_C10.pt'.format(self.cfg_trainer['percent']))

        noisylabel = noise_label['noise_label_train'][self.indexs]
        # truelabel = noise_label['clean_label_train'][self.indexs]

        self.train_labels = np.array(noisylabel)


                
            

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform_aug is not None:
            img2 = self.transform_aug(img)
        else:
            img2 = self.transform(img)


        if self.transform is not None:
            img = self.transform(img)

        

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img2, target, index, target_gt

    def __len__(self):
        return len(self.train_data)



class CIFAR10_val(torchvision.datasets.CIFAR10):

    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_val, self).__init__(root, train=train,
                                          transform=transform, target_transform=target_transform,
                                          download=download)

        # self.train_data = self.data[indexs]
        # self.train_labels = np.array(self.targets)[indexs]
        self.num_classes = 10
        self.cfg_trainer = cfg_trainer
        if train:
            self.train_data = self.data[indexs]
            self.train_labels = np.array(self.targets)[indexs]
        else:
            self.train_data = self.data
            self.train_labels = np.array(self.targets)
        self.train_labels_gt = self.train_labels.copy()
        self.indexs = indexs
        self.noise_or_not = None
    def symmetric_noise(self):
        
        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.cfg_trainer['percent'] * len(self.train_data):
                self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)

    def asymmetric_noise(self):
        for i in range(self.num_classes):
            indices = np.where(self.train_labels == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.cfg_trainer['percent'] * len(indices):
                    # truck -> automobile
                    if i == 9:
                        self.train_labels[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.train_labels[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.train_labels[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.train_labels[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.train_labels[idx] = 7
    
    def instance_noise(self):

        self.train_labels_gt = self.train_labels.copy()

        noise_label = torch.load(self.root + 'cifar-noisy/IDN_{:.1f}_C10.pt'.format(self.cfg_trainer['percent']))

        noisylabel = noise_label['noise_label_train'][self.indexs]

        self.train_labels = np.array(noisylabel)
    def __len__(self):
        return len(self.train_data)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, target_gt
        
        