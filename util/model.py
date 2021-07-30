#########################################################################
# A compact PyTorch codebase for CNN experiments
# Created by: Hammond Liu (hammond.liu@nyu.edu)
# License: GPL-3.0
# Project Url: https://github.com/hmdliu/PyTorch-CNN-Codebase/
#########################################################################

import torch.nn as nn

class Sample_Net(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(Sample_Net, self).__init__()
        self.in_feats = in_feats
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, x):
        x = x.view(-1, self.in_feats)   # [batch_size, in_feats]
        return self.linear(x)

# Original Code By: Eryk Lewinson
# Url: https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
class LeNet5(nn.Module):
    def __init__(self, in_feats, n_classes):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_feats, 6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(16, 120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, n_classes),
        )

    def forward(self, x):
        b, _, _, _ = x.size()
        feats = self.feature_extractor(x).reshape(b, -1)
        return self.classifier(feats)