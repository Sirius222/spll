from operator import index
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip, RandomErasing
from cifar import MY_CIFAR10,MY_CIFAR100, MY_CIFAR10_one_hot

np.random.seed(2)


def mycifar10_dataloaders(data_dir,rate, sup_index = None, pll_index = None, pll_label=None):
    print('Data Preparation')
    sup_train_ds = MY_CIFAR10_one_hot(data_dir, train=True, download=True,rate_partial=rate, index=sup_index, pll_label=pll_label)
    pll_train_ds = MY_CIFAR10_one_hot(data_dir, train=True, download=True,rate_partial=rate, index=pll_index, pll_label=pll_label)

    sup_train_loader = torch.utils.data.DataLoader(
        sup_train_ds,
        batch_size=64,
        # batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    pll_train_loader = torch.utils.data.DataLoader(
        pll_train_ds,
        batch_size=64,
        # batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )


    
    print('Loading dataset {0} for sup training -- Num_samples: {1}, num_classes: {2}'.format(datasets.CIFAR10.__name__,len(sup_train_ds),10))
    print('Loading dataset {0} for pll training -- Num_samples: {1}, num_classes: {2}'.format(datasets.CIFAR10.__name__,len(pll_train_ds),10))

    test_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # Data loader for test dataset
    cifar10_test_ds = datasets.CIFAR10(data_dir, transform=test_transform, train=False, download=True)
    print('Test set -- Num_samples: {0}'.format(len(cifar10_test_ds)))
    test = DataLoader(
        cifar10_test_ds, batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    return sup_train_loader, pll_train_loader, test





def cifar100_dataloaders(data_dir,rate):
    print('Data Preparation')
    cifar100_train_ds = MY_CIFAR100(data_dir,rate, sup_index = None, pll_index = None, pll_label=None)
    train_loader = torch.utils.data.DataLoader(
        cifar100_train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    print('Loading dataset {0} for training -- Num_samples: {1}, num_classes: {2}'.format(datasets.CIFAR100.__name__,len(cifar100_train_ds),100))

    test_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # Data loader for test dataset
    cifar100_test_ds = datasets.CIFAR100(data_dir, transform=test_transform, train=False, download=True)
    print('Test set -- Num_samples: {0}'.format(len(cifar100_test_ds)))
    test = DataLoader(
        cifar100_test_ds,
		batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    return train_loader, test
