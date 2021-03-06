#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install mlxtend


# In[3]:


import pandas as pd 
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder


# In[9]:


dataset=[['milk','onion','nutmeg','kidney beans','eggs','yogurt'],
        ['dill','onion','nutmeg','kidney beans','eggs','yogurt'],
        ['milk','apple','kidney beans','eggs'],
        ['milk','unicorn','corn','kidney beans','yogurt'],
        ['corn','onion','onion','kidney beans','ice cream','eggs']]


# In[10]:


te=TransactionEncoder()
te_ary=te.fit(dataset).transform(dataset)


# In[11]:


df=pd.DataFrame(te_ary,columns=te.columns_)
print(df.head())


# In[12]:


#60% minimum support
tb_df=fpgrowth(df,min_support=0.6,use_colnames=True)
print(tb_df)


# In[ ]:




