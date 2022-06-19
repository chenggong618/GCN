import pandas as pd
import networkx as nx


edges = pd.DataFrame()
# 起始节点
edges['sources'] = [1,1,1,2,2,3,3,4,4,5,5,5]
# 终止节点
edges['targets'] = [2,4,5,3,1,2,5,1,5,1,3,4]
# 点边之间的权重
edges['weights'] = [1,1,1,1,1,1,1,1,1,1,1,1]

# 定义一张图
G = nx.from_pandas_edgelist(edges,source='sources',target='targets',edge_attr='weights')
#度 /degree
print(nx.degree(G))
# 连通分量
print(nx.connected_components(G))
# 图直径（两点之间最大距离）
print(nx.diameter(G))
# 度中心性
print(nx.degree_centrality(G))
# 特征向量中心性
print(nx.eigenvector_centrality(G))
# betweenness
print(nx.betweenness_centrality(G))
# closeness
print(nx.closeness_centrality(G))
# pagerank
print(nx.pagerank(G))
# hits算法
print(nx.hits(G))


