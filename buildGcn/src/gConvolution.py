import math

import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    def __init__(self,in_features,out_features,bias=True):
        # 继承父类初始化方法
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 构造一个权重，Parameter函数将一个固定不可训练的tensor转换成可训练的类型parameter
        # 并将这个parameter绑定到这个module里面
        self.weight = Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    # 随机初始化权重
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # 从均匀分布中随机抽取并填充，这里将特征随机初始化
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self,input,adj):
        # 这里完成公式里得HW操作
        support = torch.mm(input,self.weight)
        # 这里将归一化邻接矩阵和特征权重和输入数据相乘，完成DADHW操作，完成一层GCN
        output = torch.spmm(adj,support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output