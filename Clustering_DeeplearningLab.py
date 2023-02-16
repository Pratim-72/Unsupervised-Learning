#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
file = pd.read_csv('D:/iris.csv')
file.head(10)


# In[7]:


x = file['sepal.length']
y = file['sepal.width']
plt.scatter(x,y)
plt.show()


# In[12]:


data = list(zip(x,y))
print(data)


# In[19]:


from sklearn.cluster import KMeans
model = KMeans(n_clusters = 3)
model.fit(data)
plt.scatter(x,y,c=model.labels_)
plt.show()


# In[20]:


from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters = 3)
model.fit(data)
plt.scatter(x,y,c=model.labels_)
plt.show()


# In[22]:


from scipy.cluster.hierarchy import dendrogram, linkage
linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)
plt.show()

