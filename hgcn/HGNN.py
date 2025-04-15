from torch import nn
from hgcn.layers import HGNN_conv,HGNN_classifier
# from layers import HGNN_conv,HGNN_classifier
import torch.nn.functional as F


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.2):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        # x = F.dropout(x, self.dropout)
        # x = self.hgc2(x, G)
        return x
