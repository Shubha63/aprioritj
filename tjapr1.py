#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


fh = open('groceries.csv')
maximum=0
for line in fh:
    if len(line.split())>maximum:
        maximum=len(line.split())
columns=[str(i) for i in range(0,maximum)]
columns


# In[5]:


data = pd.read_csv('groceries.csv',header=None,names=columns)
data.info()


# In[6]:


listitems = []
for i in range(0,9835):
    listitems.append([str(data.values[i,j]) for j in range(0,23)])


# In[7]:


from apyori import apriori
rules = apriori(transactions = listitems,min_support = 0.003,confidence = 0.8,min_lift = 3,min_length = 2,max_length = 2)


# In[8]:


result = list(rules)


# In[9]:


print(result)


# In[10]:


def inspect(result):
    lhs         = [tuple(r[2][0][0])[0] for r in result]
    rhs         = [tuple(r[2][0][1])[0] for r in result]
    supports    = [r[1] for r in result]
    confidences = [r[2][0][2] for r in result]
    lifts       = [r[2][0][3] for r in result]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsindata = pd.DataFrame(inspect(result), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])


# In[11]:


resultsindata


# In[12]:


resultsindata.nlargest(n=10,columns='Lift')


# In[ ]:




