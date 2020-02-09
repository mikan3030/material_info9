#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import pandas as pd
#分子データの読み込み
df = pd.read_csv("moldata_new.csv", encoding="shift-jis",skiprows=1)


# In[3]:


import networkx as nx
# m1 = Chem.MolFromSmiles("OCC1OC(C(C1O)O)(CO)OC1OC(CO)C(C(C1O)O)O")
# m2 =Chem.MolFromSmiles("ClCC1OC(C(C1O)O)(CCl)OC1OC(CO)C(C(C1O)O)Cl")
# a1 = Chem.rdmolops.GetAdjacencyMatrix(m1, useBO = True)
# a2 = Chem.rdmolops.GetAdjacencyMatrix(m2, useBO = True)
# G1 = nx.from_numpy_matrix(a1, parallel_edges=False)
# G2 = nx.from_numpy_matrix(a2, parallel_edges=False)
# label1 = dict(G1.nodes.data("atom"))



# In[64]:


G_list = []
taste = []
for i,s in zip(list(df.iloc[:,2]),list(df.iloc[:,3])):
    m1 = Chem.MolFromSmiles(i)
    try:
        a1 = Chem.rdmolops.GetAdjacencyMatrix(m1, useBO = True)
    except:
        continue
    G1 = nx.from_numpy_matrix(a1, parallel_edges=False)
    for atom in m1.GetAtoms():
        G1.nodes[atom.GetIdx()]["atom"] = atom.GetSymbol()
    if(s=="Sweet"):
        taste.append(0)
    else:
        taste.append(1)
    G_list.append(G1)


# In[65]:


def get_feature(graph):
    feature=[]
    for l1 in g.nodes(data=True):
        feature.append(l1[1]["atom"])
        ns = []
        for neighbor in g[l1[0]]:
            ns.append(g.nodes[neighbor]["atom"])
        feature.append(l1[1]["atom"]+":"+"".join(sorted(ns)))
    return feature


# In[66]:


feature_list = []
for g in G_list:
    label1 = []
    for l1 in g.nodes(data=True):
        label1.append(l1[1]["atom"])
        ns = []
        for neighbor in g[l1[0]]:
            ns.append(g.nodes[neighbor]["atom"])
        feature_list.append(l1[1]["atom"]+":"+"".join(sorted(ns)))
    feature_list.extend(label1)

feature_list=list(set(feature_list))
# feature_list


# In[67]:


import collections
import numpy as np
vec = np.zeros((len(G_list),len(feature_list)),dtype="int")
for j,g in enumerate(G_list):
    label1 = []
    for l1 in g.nodes(data=True):
        label1.append(l1[1]["atom"])
    counter = collections.Counter(get_feature(g))
    for i, k in enumerate(feature_list):
        vec[j][i] = counter[k]
#     print(counter)
# print(vec)


# In[68]:


Kernel_matrix = []
for i in range(vec.shape[0]):
    Kernel_matrix.append([0]*vec.shape[0])
    for j in range(vec.shape[0]):
        Kernel_matrix[i][j] = np.dot(vec[i],vec[j])


# In[69]:


import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA, KernelPCA
kpca = KernelPCA(n_components=2, kernel="precomputed" ,random_state=4)
X_r = kpca.fit_transform(Kernel_matrix)
plt.figure(figsize=(10,8))
colors = ["red","blue"]
for i in range(len(X_r)):
    color = colors[taste[i]]
    plt.scatter(X_r[i,0], X_r[i,1],color = color, alpha = 0.5)
plt.show()


# In[36]:


# feature_list


# In[28]:


# vec.shape


# In[43]:


# len(Kernel_matrix)


# In[ ]:




