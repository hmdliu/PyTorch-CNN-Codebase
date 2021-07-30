################################################################################################
# A compact PyTorch codebase for CNN experiments
# Modified by: Hammond Liu (hammond.liu@nyu.edu)
# Code Citation: CSCI-GA 2272-001 (Fall 2020) by Prof. Robert Fergus
# Url: https://colab.research.google.com/drive/1erzXbNGBqSaL69_gfrvKCVVH3PQHoCaG?usp=sharing
################################################################################################

import torch.nn.functional as F

def batch_classification_accuracy(output, target):
    batch_loss = F.cross_entropy(output, target, reduction='sum').item()
    pred = output.data.max(1, keepdim=True)[1]
    batch_correct = pred.eq(target.data.view_as(pred)).cpu().sum()
    return batch_correct, batch_loss