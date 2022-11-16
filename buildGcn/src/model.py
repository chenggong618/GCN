import torch.nn as nn
import torch.nn.functional as F

from buildGcn.src.gConvolution import GraphConvolution


class Gcn(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        # 继承Module的初始化方法
        super(Gcn, self).__init__()
        # 卷积操作
        self.gcn1 = GraphConvolution(nfeat, nhid)
        self.gcn2 = GraphConvolution(nhid, nhid)
        self.gcn3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    # 每层之间的传递方式
    def forward(self, x, adj):
        # relu激活函数
        x = F.relu(self.gcn1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn2(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn3(x, adj)
        return F.log_softmax(x, dim=1)
