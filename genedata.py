#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
A=nx.adjacency_matrix(G)
print(G.number_of_edges())
#print(A)
matrix=A.todense()
print(matrix)


# In[2]:


#出度
row_summs=np.sum(matrix,axis=1)
#入度
column_summs=np.sum(matrix,axis=0)
#入度的度
degrees= matrix.sum(axis=0)


# In[3]:


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


# In[4]:


#Generate possible solution vectors
def solution_vector(n,r):
    iterable=list(range(n))
    result=combinations(iterable,r)
    list_result=list(result)
    return  list_result
#Determine whether it is a solution vector
def sol_is_fvs(matrix,vector_solution):
    for i in range(0,len(vector_solution)):
            node=vector_solution[i]
            matrix[node,:]=0
            matrix[:,node]=0
    return  graph_is_ring(matrix)
#Brute force algorithm
def baoli_solution(n,matrix):
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
def bao_many(n,matrix,L):
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


matrix_test=copy.copy(matrix)
solution_node=baoli_solution(n,matrix)
print(solution_node)


# In[ ]:


L=len(solution_node)
matrix_solution=bao_many(n,matrix,L)
print(matrix_solution)

import pandas as pd
adj_matrix=np.array(matrix)
lables=np.array(matrix_solution)
#更新数据
existingdata=np.load('graphs_data.npz')
new_data={'adj_matrix_200':adj_matrix,
         'lables_200':lables}
updated_data={**existingdata,**new_data}
np.savez('graphs_data.npz',
        **updated_data)data=np.load('graphs_data.npz')
print(len(data))
#print(data['lables_50'])
