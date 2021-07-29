
from torch import nn

class Sample_Net(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(Sample_Net, self).__init__()
        self.in_feats = in_feats
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, x):
        x = x.view(-1, self.in_feats)   # [batch_size, in_feats]
        return self.linear(x)