#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
pd.read_csv(r"C:\Users\ISE2019\Downloads\dm dataset\diabetes.csv")


# In[4]:


#shape
a=pd.read_csv(r"C:\Users\ISE2019\Downloads\dm dataset\diabetes.csv")
print('shape:',a.shape)


# In[5]:


#no of columns
cols=len(a.axes[1])
print('Number of Age:',cols)


# In[6]:


#mean
m=a["Age"].mean()
print('Mean of Age:',m)


# In[11]:


#adding data
a['address']=["hyderabad,ts","Warangal,ts","Adilabad,ts","medak,ts"]
#splitting
a_split=a['address'].str.split(',',1)
a['district']=a_split.str.get(0)
a['state']=a_split.str.get(1)
del(a['address'])


# In[ ]:




