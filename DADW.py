import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import random
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import copy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

class DirectedGCNLayer(MessagePassing):
    def __init__(self, input_dim, hidden_dim):
        super(DirectedGCNLayer, self).__init__(aggr='add')
        self.linear = torch.nn.Linear(input_dim, hidden_dim)
        self.weight = torch.nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index):
        # Compute in-degree and out-degree
        row, col = edge_index
        out_deg = degree(row, x.size(0), dtype=x.dtype)
        in_deg = degree(col, x.size(0), dtype=x.dtype)

        # Normalize adjacency matrix
        out_deg_inv_sqrt = out_deg.pow(-0.5)
        in_deg_inv_sqrt = in_deg.pow(-0.5)
        norm = out_deg_inv_sqrt[row] * in_deg_inv_sqrt[col]

        # Graph convolution
        h_graph = self.propagate(edge_index, x=x, norm=norm)

        # Node feature transformation
        h_node = self.linear(x)

        # Combine
        h = h_graph + h_node
        return h

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j @ self.weight
# 前两层激活函数ReLU激活函数 最后一层激活函数Sigmoid 激活函数
#输入节点特征  图的入度和出度
#输出层特征    0到1之间的数作为概率或得分

class ThreeLayerDirectedGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ThreeLayerDirectedGCN, self).__init__()
        self.conv1 = DirectedGCNLayer(input_dim, hidden_dim)
        self.conv2 = DirectedGCNLayer(hidden_dim, hidden_dim)
        self.conv3 = DirectedGCNLayer(hidden_dim, output_dim)

    def forward(self, x, edge_index,batch):
        # First layer: ReLU activation
        h1 = self.conv1(x, edge_index)
        h1 = F.relu(h1)

        # Second layer: ReLU activation
        h2 = self.conv2(h1, edge_index)
        h2 = F.relu(h2)

        # Third layer: Sigmoid activation
        h3 = self.conv3(h2, edge_index)
        h3 = torch.sigmoid(h3)

        return h3
# 计算图的特征
def graph_features(G):
    # 计算入度和出度
    in_degrees = [G.in_degree(node) for node in G.nodes()]
    out_degrees = [G.out_degree(node) for node in G.nodes()]
    # 计算节点特征 (入度和出度)
    features = torch.tensor(list(zip(in_degrees, out_degrees)), dtype=torch.float)
    # 获取边索引 (edge_index)
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    #返回节点特征和边索引
    return features, edge_index
#训练集载入
#训练集本地地址
data = np.load('E:\graphs_data.npz')
print(len(data))
print("Arrays in the npz file:", data.files)


# In[3]:


#获得标签  提取训练集中已经得到的最小反馈节点集作为标签
def binary_label(matrix, vector):
    vector_selection = random_row = vector[np.random.randint(vector.shape[0])]
    label = np.zeros(len(matrix))
    for i in range(0, len(vector_selection)):
        loc = vector_selection[i]
        label[loc] = 1
    return label


#训练图神经网络
#数据集提取
graphs = []
for i in range(1, 200):
    matrix = data['adj_matrix_{}'.format(i)]
    vector = data['lables_{}'.format(i)]
    label = binary_label(matrix, vector)
    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)  # create_using=nx.DiGraph 表示有向图
    features, edge_index = graph_features(G)
    y = torch.tensor(label, dtype=torch.float).view(-1, 1)
    # 创建 PyTorch Geometric 数据对象
    graph_data = Data(x=features, edge_index=edge_index, y=y)
    graphs.append(graph_data)
# 使用 DataLoader 加载多个图
loader = DataLoader(graphs, batch_size=2, shuffle=True)  # 进行训练

# 创建图神经网络模型

model = ThreeLayerDirectedGCN(input_dim=2, hidden_dim=64, output_dim=1)



# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#训练模型
def train():
    model.train()
    for epoch in range(100):
        for data in loader:
            optimizer.zero_grad()

            # 获取当前批次的数据
            x, edge_index, y, batch = data.x, data.edge_index, data.y, data.batch

            # 前向传播
            output = model(x, edge_index, batch)

            # 计算损失
            loss = criterion(output, y)

            # 反向传播
            loss.backward()

            # 更新权重
            optimizer.step()

        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')


# 开始训练
train()

n = 10
p = 0.1
#生成一个有向图
G_test0 = nx.erdos_renyi_graph(n, p, directed=True)
#nx.draw(G_test0, with_labels=True, node_color='skyblue', edge_color='black')
#plt.show()
features_test0, edge_index_test0 = graph_features(G_test0)
# 测试模型（输出预测）
model.eval()
with torch.no_grad():
    x, edge_index, batch = features_test0, edge_index_test0, 1
    predictions = model(x, edge_index, batch)
    print(f'Predictions: {predictions}')



def normalize_to_ratio(data):
    total_sum = sum(data)
    return [x / total_sum for x in data]
#测试使用
#如果输出的值一样，存在过平滑问题  需要调节神经网络参数
max_value, max_index = torch.max(predictions, dim=0)
print(max_value)
print(max_index)
# 去掉多余的维度（从 [20, 1] 变为 [20]）
predictions = predictions.squeeze()
values_list = predictions.numpy().tolist()
print("预测值(列表):", values_list)
# 从大到小排序
sorted_values, sorted_indices = torch.sort(predictions, descending=True)

# 将张量转换为 Python 列表
sorted_values_list = sorted_values.tolist()
sorted_indices_list = sorted_indices.tolist()
sorted_values_list=normalize_to_ratio(sorted_values_list)

# 打印结果
print("排序后的值 (列表):", sorted_values_list)
print("对应的原位置 (列表):", sorted_indices_list)


# In[7]:


#判断有环没有
def is_directed_acyclic_graph(G):
    if nx.is_directed_acyclic_graph(G):
        return False
        #print("图是无环的")
    else:
        return True
        #print("图是有环的")


#寻找最大总和的节点
def max_total_degree_node(G):
    #获取入度和出度字典
    in_degrees = dict(G.in_degree())  # 转为字典格式
    out_degrees = dict(G.out_degree())  # 转为字典格式
    #将入度和出度相加
    total_degrees = {node: in_degrees[node] + out_degrees[node] for node in G.nodes}
    #找到最大总和的节点
    max_total_node = max(total_degrees, key=total_degrees.get)
    #max_total = total_degrees[max_node]
    return max_total_node


#基于最大总和的节点的贪婪算法
def greedy__max_total_degree_node(G):
    solution_set = []
    while is_directed_acyclic_graph(G):
        max_total_node = max_total_degree_node(G)
        G.remove_node(max_total_node)
        solution_set.append(max_total_node)
    return solution_set


#寻找最大乘积的节点
def max_product_degree_node(G):
    #获取入度和出度字典
    in_degrees = dict(G.in_degree())  # 入度字典
    out_degrees = dict(G.out_degree())  # 出度字典
    #计算每个节点的入度和出度的乘积
    degree_products = {node: in_degrees[node] * out_degrees[node] for node in G.nodes}
    #找到最大乘积的节点
    max_product_node = max(degree_products, key=degree_products.get)
    #max_product = degree_products[max_node]
    return max_product_node


#基于最大乘积的节点的贪婪算法
def greedy__max_product_degree_node(G):
    solution_set = []
    while is_directed_acyclic_graph(G):
        max_product_node = max_product_degree_node(G)
        G.remove_node(max_product_node)
        solution_set.append(max_product_node)
    return solution_set


#寻找出度和入度中最大的度的节点
def max__value__degree_node(G):
    #获取入度和出度字典
    in_degrees = dict(G.in_degree())  # 入度字典
    out_degrees = dict(G.out_degree())  # 出度字典
    #计算每个节点的入度和出度中的最大值
    max_degrees = {node: max(in_degrees[node], out_degrees[node]) for node in G.nodes}
    #找出度和入度中最大的度的节点
    max_value_node = max(max_degrees, key=max_degrees.get)
    #max_value = max_degrees[max_value_node]
    return max_value_node


#基于出度和入度中最大的度的节点的贪婪算法
def greedy__max__value__degree_node(G):
    solution_set = []
    while is_directed_acyclic_graph(G):
        max_value_node = max__value__degree_node(G)
        G.remove_node(max_value_node)
        solution_set.append(max_value_node)
    return solution_set


# In[8]:

#测试图
n = 100
p = 0.15
#生成一个有向图
G_test = nx.erdos_renyi_graph(n, p, directed=True)
edge_count = G_test.number_of_edges()
print(f":边的数目")
print("有向边的数目:", edge_count)
#nx.draw(G_test, with_labels=True, node_color='skyblue', edge_color='black')
#plt.show()
features_test, edge_index_test = graph_features(G_test)
# In[9]:



#测试算法
#浅拷贝
G_test0 = copy.copy(G_test)
#深拷贝


G_test1 = copy.deepcopy(G_test)
solution_1 = greedy__max_total_degree_node(G_test1)
sorted_solution_1 = sorted(solution_1)
print(f":基于最大总和的节点的贪婪算法")
print(len(solution_1))
print(sorted_solution_1 )
print(f":基于最大乘积的节点的贪婪算法")
G_test2 = copy.deepcopy(G_test)
solution_2 = greedy__max_product_degree_node(G_test2)
sorted_solution_2 = sorted(solution_2)
print(len(solution_2))
print(sorted_solution_2)
print(f":基于出度和入度中最大的度的节点的贪婪算法")
G_test3 = copy.deepcopy(G_test)
solution_3 = greedy__max__value__degree_node(G_test3)
sorted_solution_3 = sorted(solution_3)
print(len(solution_3))
print(sorted_solution_3)
# In[10]:






# GNN输出的节点特征大小排序
def GNN_predictions_node(G, model):
    features, edge_index = graph_features(G)
    model.eval()
    with torch.no_grad():
        #通过GNN预测
        x, edge_index, batch = features, edge_index, torch.tensor([0] * features.size(0), dtype=torch.long)
        #GNN网络输出预测结果
        predictions = model(x, edge_index, batch)
        #结果排序
        # 去掉多余的维度（从 [20, 1] 变为 [20]）
        predictions = predictions.squeeze()
        # 从大到小排序
        sorted_values, sorted_indices = torch.sort(predictions, descending=True)
        # 将张量转换为 Python 列表
        sorted_values_list = sorted_values.tolist()
        sorted_indices_list = sorted_indices.tolist()
        sorted_values_list=normalize_to_ratio(sorted_values_list)
        return sorted_indices_list, sorted_values_list

# 贪婪算法：每次删除所得特征对应的值最大的节点，直到图没有环
def greedy_selection_node(G, model):
    #作为解集
    solution_set = []
    # GNN输出的节点特征大小排序
    sorted_indices_list, sorted_values_list=GNN_predictions_node(G, model)
    #贪婪选择
    for i in range (len(sorted_indices_list)):
        selection_node=sorted_indices_list[i]
        #print(f"删除节点: {selection_node}, 概率值: {sorted_values_list[i]}")
        G.remove_node(selection_node)
        solution_set.append(selection_node)  # 记录最大概率节点
        if is_directed_acyclic_graph(G)==False:
                break
    return solution_set



# In[11]:
#贪婪算法 不迭代
G_test4 = copy.deepcopy(G_test)
model.eval()
solution_4 = greedy_selection_node(G_test4, model)
sorted_solution_4 = sorted(solution_4)
print(f":基于GNN的贪婪算法")
print(len(solution_4))
print(sorted_solution_4 )


#贪婪算法 迭代
#通过邻接矩阵删除节点
def remove_node(G, max_pro_node):
    A=nx.adjacency_matrix(G)
    adj_matrix=A.todense()
    adj_matrix[max_pro_node,:]=0
    adj_matrix[:,max_pro_node]=0
    return adj_matrix
#避免重复选择
def ite_selection_node(sorted_indices_list, sorted_values_list, solution_set):
     for num in sorted_indices_list:  # 遍历排序好的数组
        if num not in solution_set:  # 如果当前节点不在解集中，则选择该节点
            max_pro_node=num
            index = sorted_indices_list.index(num)
            max_value_node=sorted_values_list[index]
            return max_pro_node,max_value_node  # 返回该元素
#迭代贪婪算法 删除一个节点，重新通过GNN
def greedy_ite_selection_node(G, model):
    solution_set = []
    while is_directed_acyclic_graph(G):  # 图有环时继续删除
        sorted_indices_list, sorted_values_list=GNN_predictions_node(G, model)
        max_pro_node,max_value_node=ite_selection_node(sorted_indices_list, sorted_values_list, solution_set)
        adj_matrix= remove_node(G, max_pro_node)
        G=nx.from_numpy_array(adj_matrix,create_using=nx.DiGraph())
        solution_set.append(max_pro_node)  # 记录最大概率节点
    return solution_set


#调用贪婪算法 迭代
G_test5 = copy.deepcopy(G_test)
model.eval()
solution_5 = greedy_ite_selection_node(G_test5, model)
sorted_solution_5 = sorted(solution_5)
print(f":基于GNN迭代的贪婪算法")
print(len(solution_5))
print(sorted_solution_5)



#波束搜索


# 避免重复选择
def beam_selection_node(sorted_indices_list, sorted_values_list, solution_set, beam_width):
    candidates = []
    for num in sorted_indices_list:  # 遍历排序好的数组
        if num not in solution_set:  # 如果当前节点不在解集中，则选择该节点
            index = sorted_indices_list.index(num)
            value = sorted_values_list[index]
            candidates.append((num, value))
            if len(candidates) >= beam_width:
                break
    return candidates


# 波束搜索算法
def beam_search_selection_node(G, model, beam_width=6, alpha=0.5):
    # 初始化候选解集合，每个候选解是一个 (solution_set, graph, score) 的元组
    candidates = [([], G, 0.0)]  # 初始候选解：空解集、原始图、初始评分

    while True:
        new_candidates = []
        for solution_set, current_G, score in candidates:
            if nx.is_directed_acyclic_graph(current_G):  # 如果当前图已经无环，跳过
                new_candidates.append((solution_set, current_G, score))
                continue

            # 获取当前图的预测结果
            sorted_indices_list, sorted_values_list = GNN_predictions_node(current_G, model)
            if hasattr(sorted_values_list, 'tolist'):
                sorted_values_list = sorted_values_list.tolist()  # 转换为 Python 列表

            # 选择 beam_width 个候选节点
            node_candidates = beam_selection_node(sorted_indices_list, sorted_values_list, solution_set, beam_width)

            # 生成新的候选解
            for node, value in node_candidates:
                new_solution_set = solution_set.copy()
                new_solution_set.append(node)  # 添加新节点到解集
                new_adj_matrix = remove_node(current_G, node)  # 删除节点
                new_G = nx.from_numpy_array(new_adj_matrix, create_using=nx.DiGraph())  # 创建新图

                # 计算新的评分（累加节点值）
                new_score = score + value

                # 对评分进行长度归一化
                normalized_score = new_score / (len(new_solution_set) ** alpha)

                new_candidates.append((new_solution_set, new_G, normalized_score))

        # 如果所有候选解都已经无环，退出循环
        if all(nx.is_directed_acyclic_graph(g) for _, g, _ in new_candidates):
            break

        # 保留归一化评分最高的 beam_width 个候选解
        candidates = sorted(new_candidates, key=lambda x: x[2], reverse=True)[:beam_width]

    # 打印所有候选解
    print("所有候选解：")
    for i, (solution_set, _, score) in enumerate(candidates, start=1):
        print(f"候选解 {i}: 删除节点 {solution_set}, 归一化评分: {score:.4f}")

    # 选择删除节点个数最小的候选解
    min_length_solution = min(candidates, key=lambda x: len(x[0]))
    print(f"\n最小删除节点个数的候选解：删除节点 {min_length_solution[0]}")

    # 返回最小长度的候选解
    return min_length_solution[0]
'''
def beam_search_selection_node(G, model, beam_width=10):
    # 初始化候选解集合，每个候选解是一个 (solution_set, graph) 的元组
    candidates = [([], G)]  # 初始候选解：空解集和原始图

    while True:
        new_candidates = []
        for solution_set, current_G in candidates:
            if nx.is_directed_acyclic_graph(current_G):  # 如果当前图已经无环，跳过
                new_candidates.append((solution_set, current_G))
                continue

            # 获取当前图的预测结果
            sorted_indices_list, sorted_values_list = GNN_predictions_node(current_G, model)
            if hasattr(sorted_values_list, 'tolist'):
                sorted_values_list = sorted_values_list.tolist()  # 转换为 Python 列表

            # 选择 beam_width 个候选节点
            node_candidates = beam_selection_node(sorted_indices_list, sorted_values_list, solution_set, beam_width)

            # 生成新的候选解
            for node, value in node_candidates:
                new_solution_set = solution_set.copy()
                new_solution_set.append(node)  # 添加新节点到解集
                new_adj_matrix = remove_node(current_G, node)  # 删除节点
                new_G = nx.from_numpy_array(new_adj_matrix, create_using=nx.DiGraph())  # 创建新图
                new_candidates.append((new_solution_set, new_G))

        # 如果所有候选解都已经无环，退出循环
        if all(nx.is_directed_acyclic_graph(g) for _, g in new_candidates):
            break

        # 保留概率值最高的 beam_width 个候选解
        candidates = sorted(new_candidates, key=lambda x: sum(sorted_values_list), reverse=True)[:beam_width]

    # 打印所有候选解
    print("所有候选解：")
    for i, (solution_set, _) in enumerate(candidates, start=1):
        print(f"候选解 {i}: 删除节点 {solution_set}")

    # 选择删除节点个数最小的候选解
    min_length_solution = min(candidates, key=lambda x: len(x[0]))
    print(f"\n最小删除节点个数的候选解：删除节点 {min_length_solution[0]}")

    # 返回最小长度的候选解
    return min_length_solution[0]
'''
G_test6 = copy.deepcopy(G_test)
model.eval()
solution_6 = beam_search_selection_node(G_test6, model)
sorted_solution_6 = sorted(solution_6)
print(f":基于波束搜索的启发式算法")
print(len(solution_6))
print(sorted_solution_6 )

def beam_search_selection_rand_node(G, model, beam_width=6, randomness=0.3):
    # 初始化候选解集合，每个候选解是一个 (solution_set, graph) 的元组
    candidates = [([], G)]  # 初始候选解：空解集和原始图

    while True:
        new_candidates = []
        for solution_set, current_G in candidates:
            if nx.is_directed_acyclic_graph(current_G):  # 如果当前图已经无环，跳过
                new_candidates.append((solution_set, current_G))
                continue

            # 获取当前图的预测结果
            sorted_indices_list, sorted_values_list = GNN_predictions_node(current_G, model)
            if hasattr(sorted_values_list, 'tolist'):
                sorted_values_list = sorted_values_list.tolist()  # 转换为 Python 列表

            # 选择 beam_width 个候选节点
            node_candidates = beam_selection_node(sorted_indices_list, sorted_values_list, solution_set, beam_width)

            # 生成新的候选解
            for node, value in node_candidates:
                new_solution_set = solution_set.copy()
                new_solution_set.append(node)  # 添加新节点到解集
                new_adj_matrix = remove_node(current_G, node)  # 删除节点
                new_G = nx.from_numpy_array(new_adj_matrix, create_using=nx.DiGraph())  # 创建新图
                new_candidates.append((new_solution_set, new_G))

        # 如果所有候选解都已经无环，退出循环
        if all(nx.is_directed_acyclic_graph(g) for _, g in new_candidates):
            break

        # 保留概率值最高的 beam_width 个候选解，但加入随机化选择
        candidates = stochastic_beam_selection(new_candidates, sorted_values_list, beam_width, randomness)

    # 打印所有候选解
    print("所有候选解：")
    for i, (solution_set, _) in enumerate(candidates, start=1):
        print(f"候选解 {i}: 删除节点 {solution_set}")

    # 选择删除节点个数最小的候选解
    min_length_solution = min(candidates, key=lambda x: len(x[0]))
    print(f"\n最小删除节点个数的候选解：删除节点 {min_length_solution[0]}")

    # 返回最小长度的候选解
    return min_length_solution[0]

def stochastic_beam_selection(candidates, sorted_values_list, beam_width, randomness):
    # 首先按照概率值排序
    sorted_candidates = sorted(candidates, key=lambda x: sum(sorted_values_list), reverse=True)

    # 保留概率值最高的 beam_width 个候选解
    top_candidates = sorted_candidates[:beam_width]

    # 以一定的概率随机选择其他候选解
    if random.random() < randomness:
        # 从剩余的候选解中随机选择一个
        remaining_candidates = sorted_candidates[beam_width:]
        if remaining_candidates:
            random_candidate = random.choice(remaining_candidates)
            # 替换掉 top_candidates 中的一个随机候选解
            replace_index = random.randint(0, beam_width - 1)
            top_candidates[replace_index] = random_candidate

    return top_candidates
G_test7 = copy.deepcopy(G_test)
model.eval()
solution_7 = beam_search_selection_rand_node(G_test7, model)
sorted_solution_7 = sorted(solution_7)
print(f":基于随机波束搜索的启发式算法")
print(len(sorted_solution_7))
print(sorted_solution_7)