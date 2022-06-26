import torch.nn as nn
import torch.nn.functional as F

from pygcn.pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        #构造第一层GCN--第一个参数是初始的特征，第二个参数是隐藏层的一个特征
        self.gc1 = GraphConvolution(nfeat, nhid)
        #构造第二层GCN
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
