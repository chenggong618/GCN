import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    # 读取data文件夹中后缀为.content的文件
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    #截取索引为1到倒数第一的数据作为特征向量，倒数第一位不截取--python的特性
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    #截取索引为倒数第一位作为标签
    labels = encode_onehot(idx_features_labels[:, -1])
    """build graph--构建图"""
    #获取数据索引
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    #将顶点从0开始做索引并且一一对应
    idx_map = {j: i for i, j in enumerate(idx)}
    #读取data文件夹当中后缀名为.cites的数据--文件里保存的边数据
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    #将顶点转换成用idx_map当中存储的索引表示
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    #构建边的邻接矩阵--很重要
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix--计算转置矩阵并且构造一个堆成矩阵，这一步相当于把一个有向图转换成无向图
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #特征归一化
    features = normalize(features)
    #对A+I归一化
    adj = normalize(adj + sp.eye(adj.shape[0]))

    #设置训练、验证、测试集
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    #将numpy的数据转换成torch的数据
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    #矩阵的行求和
    rowsum = np.array(mx.sum(1))
    #求倒数
    r_inv = np.power(rowsum, -1).flatten()
    #将无穷大数变为0
    r_inv[np.isinf(r_inv)] = 0.
    #r_inv变成对角矩阵
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
