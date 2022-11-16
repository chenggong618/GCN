import argparse
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from buildGcn.src.model import Gcn
from buildGcn.src.utils import load_data, accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
# 判断CUDA是否可用
print(torch.cuda.is_available())
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 设置随机数，seed()函数的用法是为了保证每次产生的随机数都一致方便下次实验
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# 如果CUDA可用则设置关于GPU的随机数并且每次实验保持一致
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# 加载并处理数据
adj, labels, features, idx_train, idx_val, idx_test = load_data("../data/cora/", "cora")
# 初始化两层GCN模型
model = Gcn(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout)
# 设置训练时优化算法，目的是为了加快训练速度
optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# 如果GPU存在则把模型从cpu转换成GPU，XX.cuda()就是转换方法
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    idx_train = idx_train.cuda()


# 训练
def train(epoch):
    t = time.time()
    model.train()
    optimiser.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimiser.step()

    if not args.fastmode:
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    # 输出训练结果
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    # 输出测试结果
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

test()
