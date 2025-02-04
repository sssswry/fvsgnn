#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import queue 
import copy
from itertools import combinations
#参数设置 Parameter settings
n=20
p=0.1
G=nx.erdos_renyi_graph(n,p,directed=True)
nx.draw(G, with_labels=True, node_color='skyblue',
        edge_color='black')
plt.show()
#图的邻接矩阵
A=nx.adjacency_matrix(G)
print(G.number_of_edges())
#print(A)
matrix=A.todense()
print(matrix)


# In[2]:


#出度 Out-degree
out_degree =np.sum(matrix,axis=1)
#入度 in_degree
in_degree=np.sum(matrix,axis=0)
#入度的度
#degrees= matrix.sum(axis=0)


# In[3]:


#判断有环没有 通过拓扑排序
import queue
def is_directed_acyclic_graph(matrix):
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


# In[4]:


#Generate possible solution vectors
#通过排列组合 遍历所有可能潜在存在的解
def solution_vector(n,r):
    iterable=list(range(n))
    result=combinations(iterable,r)
    list_result=list(result)
    return  list_result
#Determine whether it is a solution vector
#根据有没有环，从潜在存在的解向量长度从小到大开始探索
#从而找到最小的解
def sol_is_fvs(matrix,vector_solution):
    for i in range(0,len(vector_solution)):
            node=vector_solution[i]
            matrix[node,:]=0
            matrix[:,node]=0
    return  is_directed_acyclic_graph(matrix)
#Brute force algorithm
#暴力算法求解
def brute_force_solution(n,matrix):
    solution_node=[]
    for i in range(0,n):
        list_result=solution_vector(n,i)
        for j in range (0,len(list_result)):
            matrix_test=copy.copy(matrix)
            vector_solution=list(list_result[j])
            flag= sol_is_fvs(matrix_test,vector_solution)
            if flag==False:
                #solution_node=vector_solution
                return vector_solution
               # return solution_node
               # stop=True
              #  break
       # if stop:
          #  break
   # return solution_node


# In[5]:


# Brute force algorithm for many solutions
#根据上面找的最小长度的解，遍历其他同样长度下潜在的解
def many_solutions(n,matrix,L):
    solution_node=[]
    matrix_solution=[]
    list_result=solution_vector(n,L)
    for j in range (0,len(list_result)):
        matrix_test=copy.copy(matrix)
        vector_solution=list(list_result[j])
        flag= sol_is_fvs(matrix_test,vector_solution)
        if flag==False:
            matrix_solution.append(vector_solution)
                #solution_node=vector_solution
           # print(vector_solution)
               # return solution_node
               # stop=True
              #  break
       # if stop:
          #  break
    return matrix_solution


# In[ ]:

#输出解
matrix_test=copy.copy(matrix)
solution_node=brute_force_solution(n,matrix)
print(solution_node)


# In[ ]:

#输出解
L=len(solution_node)
matrix_solution=many_solutions(n,matrix,L)
print(matrix_solution)

adj_matrix=np.array(matrix)
lables=np.array(matrix_solution)
#存入更新数据
existingdata=np.load('graphs_data.npz')
new_data={'adj_matrix_200':adj_matrix,
         'lables_200':lables}
updated_data={**existingdata,**new_data}
np.savez('graphs_data.npz',**updated_data)
data=np.load('graphs_data.npz')
print(len(data))
#print(data['lables_50'])
