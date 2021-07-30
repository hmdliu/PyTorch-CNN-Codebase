#########################################################################
# A compact PyTorch codebase for CNN experiments
# Created by: Hammond Liu (hammond.liu@nyu.edu)
# License: GPL-3.0
# Project Url: https://github.com/hmdliu/PyTorch-CNN-Codebase/
#########################################################################

from torchvision import datasets, transforms

def get_mnist(train_flag):
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return datasets.MNIST(
            root='./', 
            train=train_flag, 
            download=True, 
            transform=data_transform
        )

def get_cifar10(train_flag):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return datasets.CIFAR10(
        root='./', 
        train=train_flag, 
        download=True, 
        transform=data_transform
    )