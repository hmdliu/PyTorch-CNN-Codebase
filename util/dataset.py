################################################################################################
# A compact PyTorch codebase for CNN experiments
# Modified by: Hammond Liu (hammond.liu@nyu.edu)
# Code Citation: CSCI-GA 2272-001 (Fall 2020) by Prof. Robert Fergus
# Url: https://colab.research.google.com/drive/1erzXbNGBqSaL69_gfrvKCVVH3PQHoCaG?usp=sharing
################################################################################################

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