import numpy as np
import scipy.sparse as sp
import torch


def load_data(path="", dataset=""):
    print('Loading {} dataset...'.format(dataset))
    # 读取data文件夹当中cora.content文件里的内容
    cora_contents = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    # 读取content文件当中1到倒数第二行内容作为特征矩阵。csr_matrix采用按行压缩的办法用三个数组表示原来的稀疏矩阵
    features = sp.csr_matrix(cora_contents[:, 1:-1], dtype=np.float32)
    # features = features.todense() # 稀疏矩阵还原
    # 获取最后一位作为标签
    labels = encode_onehot(cora_contents[:, -1])
    # 读取content文件的第一列，这一列代表节点的编号
    idx = np.array(cora_contents[:, 0], dtype=np.int32)
    idx_dict = {j: i for i, j in enumerate(idx)}

    cora_cites = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    # flatten函数的意义是给数据降维
    edges = np.array(list(map(idx_dict.get, cora_cites.flatten())), dtype=np.int32).reshape(cora_cites.shape)
    # 构建邻接矩阵，coo_matrix函数构造函数参数指定的矩阵,在这里构建一个 labels.shape大小的方阵，每条边所在的位置为1
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # 这套骚操作是为了创建无向图的邻接矩阵，存储方式是散列表的形式
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 归一化操作
    features = normalize(features)
    # 这里完成卷积公式里面的DAD操作
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = torch.LongTensor(range(140))
    idx_val = torch.LongTensor(range(200, 500))
    idx_test = torch.LongTensor(range(500, 1500))
    # 将矩阵转换成tensor结构
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    # 将稀疏矩阵转换成tensor数据结构
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, labels, features, idx_train, idx_val, idx_test


def encode_onehot(labels):
    # set挑选出所有分类标签
    classes = set(labels)
    # enumerate将序列数组当中元素和索引遍历出来
    # 这里将分类内容组成字典，相当于把标签映射为向量
    class_matrix =  np.identity(len(classes))[1,:]
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


# 将矩阵归一化的方法
def normalize(mx):
    """Row-normalize sparse matrix"""
    # 矩阵的行求和
    rowsum = np.array(mx.sum(1))
    # 求倒数
    r_inv = np.power(rowsum, -1).flatten()
    # 将无穷大数变为0
    r_inv[np.isinf(r_inv)] = 0.
    # r_inv变成对角矩阵
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# 稀疏矩阵转换成tensor数据结构方法
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# 计算准确度
def accuracy(output, labels):
    # type_as 类型转换函数
    preds = output.max(1)[1].type_as(labels)
    # 对比预测结果和测试集结果
    correct = preds.eq(labels).double()
    correct = correct.sum()
    # 计算准确度并返回
    return correct / len(labels)
