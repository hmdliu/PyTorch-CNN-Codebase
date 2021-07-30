#########################################################################
# A compact PyTorch codebase for CNN experiments
# Created by: Hammond Liu (hammond.liu@nyu.edu)
# License: GPL-3.0
# Project Url: https://github.com/hmdliu/PyTorch-CNN-Codebase/
#########################################################################

import torch.nn.functional as F

def batch_classification_accuracy(output, target):
    batch_loss = F.cross_entropy(output, target, reduction='sum').item()
    pred = output.data.max(1, keepdim=True)[1]
    batch_correct = pred.eq(target.data.view_as(pred)).cpu().sum()
    return batch_correct, batch_loss