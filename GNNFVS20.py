#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch_geometric.utils as utils
from torch_geometric.nn import GCNConv
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


# In[2]:


data=np.load('graphs_data.npz')
print(len(data))


# In[3]:


def binary_label(matrix,vector):
    vector_selection=random_row = vector[np.random.randint(vector.shape[0])]
    label=np.zeros(len(matrix))
    for i in range(0,len(vector_selection)):
        loc= vector_selection[i]
        label[loc]=1 
    return label


# In[4]:


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
class GCNConv(nn.Module):
    def __init__(self, A, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.A_hat = A+torch.eye(A.size(0))
        self.D     = torch.diag(torch.sum(self.A_hat,1))
        self.D     = self.D.inverse().sqrt()
        self.A_hat = torch.mm(torch.mm(self.D, self.A_hat), self.D)
        self.W     = nn.Parameter(torch.rand(in_channels,out_channels, requires_grad=True))
    def forward(self, X):
        out = torch.relu(torch.mm(torch.mm(self.A_hat, X), self.W))
        return torch.sigmoid(out)

class Net(torch.nn.Module):
    def __init__(self,A, nfeat, nhid, nout):
        super(Net, self).__init__()
        self.conv1 = GCNConv(A,nfeat, nhid,)
        self.conv2 = GCNConv(A,nhid, nout)        
    def forward(self,X):
        H  = self.conv1(X)
        H2 = self.conv2(H)
        return H2


# In[5]:


for i in range (1,50):
        
    matrix=data['adj_matrix_{}'.format(i)]
    vector=data['lables_{}'.format(i)]
    label=binary_label(matrix,vector)

    A=torch.Tensor(matrix)
    target=torch.LongTensor(label)
    X=torch.eye(A.size(0))
    T=Net(A,X.size(0), 64, 2)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(T.parameters(), lr=0.01)
    total_loss =0  
    for j in range(200):
        optimizer.zero_grad()
        output = T(X)
        loss=criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if j %200==0:
             print("Cross Entropy Loss: =", loss.item())


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import queue 
import copy
from itertools import combinations
#参数设置 Parameter settings
n=500
p=0.3
G=nx.erdos_renyi_graph(n,p,directed=True)
nx.draw(G, with_labels=True, node_color='skyblue',
        edge_color='black')
plt.show()
A=nx.adjacency_matrix(G)
matrix_test=A.todense()


# In[ ]:


#判断有环没有
import queue
def graph_is_ring(matrix):
    n=len(matrix)
    q=queue.Queue()
    visited=[]
    degrees= matrix.sum(axis=0)
    for i in range (0,n):
        if degrees[i]==0:
            q.put(i)
            visited.append(i)
    while not q.empty():
        i=q.get()
        for j in range(0,n):
            if int(matrix[i][j])==1:
                degrees[j]-=1
                if degrees[j]==0:
                    q.put(j)
                    visited.append(j)
    if len(visited)==n:
         return False
    else:
         return  True


# In[ ]:


#贪婪算法
def find_max_sum_degree(matrix):
    degrees= matrix.sum(axis=0)+np.sum(matrix,axis=1)
    max_degrees = max(degrees)
    for i in range (len(matrix)):
        if degrees[i] == max_degrees:
            loc=i
    return loc

def greedy_solution_sum(matrix):
    visited_node=[]
    while graph_is_ring(matrix) ==True:
        node=find_max_sum_degree(matrix)
        matrix[node,:]=0
        matrix[:,node]=0
        visited_node.append(node)
    return  visited_node
def find_max_mul_degree(matrix):
    row_summs=np.sum(matrix,axis=1)
    column_summs=np.sum(matrix,axis=0)
    degrees=[x*y for x,y in zip(row_summs,column_summs)]
    max_degrees = max(degrees)
    for i in range (len(matrix)):
        if degrees[i] == max_degrees:
            loc=i
    return loc

def greedy_solution_mul(matrix):
    visited_node=[]
    while graph_is_ring(matrix) ==True:
        node=find_max_mul_degree(matrix)
        matrix[node,:]=0
        matrix[:,node]=0
        visited_node.append(node)
    return  visited_node


# In[ ]:


#贪婪算法
def find_max(matrix):
    row_summs=np.sum(matrix,axis=1)
    column_summs=np.sum(matrix,axis=0)
    degrees=[0]*len(row_summs)
    for i in range(len(row_summs)):
        if row_summs[i]> column_summs[i]:
              degrees[i]=row_summs[i]
        else:
            degrees[i]=column_summs[i]
    return degrees

def find_max_degree(matrix):
    degrees= find_max(matrix)
    max_degrees = max(degrees)
    for i in range (len(matrix)):
        if degrees[i] == max_degrees:
            loc=i
    return loc

def greedy_solution_max(matrix):
    visited_node=[]
    while graph_is_ring(matrix) ==True:
        node=find_max_degree(matrix)
        matrix[node,:]=0
        matrix[:,node]=0
        visited_node.append(node)
    return  visited_node


# In[ ]:


matrix_test1=copy.copy(matrix_test)
matrix_test2=copy.copy(matrix_test)
matrix_test3=copy.copy(matrix_test)
solution_sum=greedy_solution_sum(matrix_test1)
solution_mul=greedy_solution_mul(matrix_test2)
solution_max=greedy_solution_max(matrix_test3)
print(len(solution_sum))
print(len(solution_mul))
print(len(solution_max))
#print(solution_sum)
#print(solution_mul)
#print(solution_max)


# In[ ]:


# 根据值排序，选择前n个最大的值
def find_largest_n_indices(lst, n):
    indexed_lst = list(enumerate(lst))
    largest_n = sorted(indexed_lst, key=lambda x: x[1], reverse=True)[:n]
    return [index for index, value in largest_n]


# In[ ]:


import math
def integer_sqrt(x):
    return math.floor(math.sqrt(x))


# In[ ]:


#gnn评估
def gnn_max(matrix):
    k=integer_sqrt(matrix.shape[0])
    A_test=torch.Tensor(matrix)
    X_test=torch.eye(A_test.size(0))
    T=Net(A_test,X_test.size(0), 10, 1)
    prob=[]
    for i in range(0,len(matrix)):
        prob.append(T(A_test)[i,0].item())
    max_index_rank=find_largest_n_indices(prob, k)
    return max_index_rank


# In[ ]:


#GNN选择策略
def greedy_solution_gnn(matrix,i):
    visited_node=[]
    while graph_is_ring(matrix) ==True:
        node_rank=gnn_max(matrix)
        node=node_rank[i]
        matrix[node,:]=0
        matrix[:,node]=0
        visited_node.append(node)
    return  visited_node


# In[ ]:


#最小反馈顶点集
def min_gnn_solution(matrix,k):
    min_solution=matrix_test0.shape[0];
    for i in range (0,k):
        solution_gnn=greedy_solution_gnn(matrix,i)
        solution_gnn=np.unique(solution_gnn)
        len_solution=len(solution_gnn)
        if len_solution<min_solution:
            min_solution=len_solution
        return min_solution


# In[ ]:


#GNN
import copy
matrix_test0=copy.copy(matrix_test)
k=10
min_solution=min_gnn_solution(matrix_test0,k);
print(min_solution)


# In[ ]:


print(len(solution_sum))
print(len(solution_mul))
print(len(solution_max))
print(min_solution)

