#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[1]:


pip install apriori


# In[3]:


pip install apyori


# In[5]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
from apyori import apriori
store_data=pd.read_csv(r"C:\Users\ISE2019\Downloads\dm dataset\diabetes.csv" ,low_memory=False,header=None )
print("first two records of database\n\n", store_data.head(2))
print("\n\n number of rows and columns", store_data.shape) 
print("\n\n")
records=[]
for i in range (0.7501):
 records.append([str(store_data.values[i,j]) for j in range(0.20)])
 records [0]
association_rules=apriori (records, min_suppose=0.045,min_confidence=0.2,min_lift=3,min_length=2) 
results=list (association_rules)
print("first rule in list or results\n", results[0]) 
print("\n")


# In[ ]:




