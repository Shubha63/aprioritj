#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
a=np.array([[1,2,3,],[2,3,4],[3,4,5]])
b=a.shape
print("shape:",a.shape)
c=a.ndim
print("dimensions:",a.ndim)


# In[5]:


z=np.zeros((2,2))
print("zeros:",z)
o=np.ones((2,2))
print("ones:",o)


# In[14]:


a=np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]])
b=a.reshape(1,4,4)
print("reshape:",b)
c=a.flatten()
print("flatten:",c)


# In[7]:


x=np.array([[10,20],[60,70]])
y=np.array([[30,40],[80,90]])
v=np.vstack((x,y))
print("vertically:",v)
h=np.hstack((x,y))
print("horizontally:",h)


# In[15]:


a=np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]])
temp=a[[0,1,2,3],[3,2,1,0]]
print("indexing:",temp)
i=a[:4, ::2]
print("slicing:",i)


# In[10]:


a=np.array([[1,3,-1,4],[3,-2,1,4]])
b=a.min()
print("minimum:",b)
c=a.max()
print("maximum:",c)
a=np.array([1,2,3,4,5])
d=a.mean()
print("mean:",d)
e=np.median(a)
print("median:",e)
f=a.std()
print("standard deviation:",f)


# In[2]:


import pandas as pd
pd.read_csv(r"C:\Users\ISE2019\Desktop\dm dataset\dia-small.csv")


# In[3]:


a=pd.read_csv(r"C:\Users\ISE2019\Desktop\dm dataset\dia-small.csv")
print('shape:',a.shape)


# In[4]:


cols=len(a.axes[1])
print('no of columns:',cols)


# In[5]:


m=a["Age"].mean()
print('mean of age:',m)


# In[21]:


a['address']=["hyderbad,ts","warangal,ts","adilabad,ts","medak,ts"]
a_split=a['address'].str.split(',',1)
a['district']=a_split.str.get(0)
a['state']=a_split.str.get(1)
del(a['address'])


# In[27]:


import matplotlib as plt
a.plot.scatter(x='Glucose',y='BMI',c='SkinThickness')


# In[37]:


import pandas as pd
dia_data=pd.read_csv(r"C:\Users\ISE2019\Desktop\dm dataset\dia-small.csv")
records=len(dia_data)
print("display first 10 records",records)
dia_data.head(10)


# In[39]:


import pandas as pd
import numpy as np
a1=pd.Series([2,3,4])
a1.describe()


# In[44]:


import pandas as pd
import numpy as np
dia_data=pd.read_csv(r"C:\Users\ISE2019\Desktop\dm dataset\data1.csv")
print(dia_data.shape)
dia_data.dropna(inplace=True)
print(dia_data.shape)


# In[1]:


import pandas as pd
import numpy as np
dia_data=pd.read_csv(r"C:\Users\ISE2019\Desktop\dm dataset\data1.csv")
print("Before use of dropna()",dia_data.shape)
dia_data.dropna(axis=1,how="all",inplace=True)
print("After use of dropna()",dia_data.shape)
dia_data.head(10)


# In[57]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from apyori import apriori
st_df = pd.read_csv(r"C:\Users\ISE2019\Desktop\dm dataset\store_data.csv",low_memory=False,header=None)

print("First two records of data base\n \n",st_df.head(2))

print("\n \n Number of rows and columns",st_df.shape)
print("\n\n")

records = []
for i in range(7501):
    records.append([str(st_df.values[i,j]) for j in range(20)])
records[0]

association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)

print("First rule in list \n",association_results[0])
print("\n")


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from apyori import apriori
st_df = pd.read_csv(r"C:\Users\ISE2019\Desktop\dm dataset\bank-marketing_csv.csv",low_memory=False,header=None)

print("First two records of data base\n \n",st_df.head(2))

print("\n \n Number of rows and columns",st_df.shape)
print("\n\n")

records = []
for i in range(10):
    records.append([str(st_df.values[i,j]) for j in range(6)])
records[0]

association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)

print("First rule in list \n",association_results[0])
print("\n")


# In[2]:


import pandas as pd
import numpy as np
from apyori import apriori


# In[3]:


items = [['A','B'],['B','D'],['B','C'],['A','B','D'],['A','C'],['B','C'],['A','C'],['A','B','C','E'],['A','B','C']]
items_data = pd.DataFrame(items)
items_data


# In[4]:


items_data.info()


# In[5]:


rules=apriori(transactions=items,min_support=0.003,confidence=0.5)
result=list(rules)
result[10]


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from apyori import apriori
st_df = pd.read_csv(r"C:\Users\ISE2019\Desktop\dm dataset\itemset.csv",low_memory=False,header=None)

print("First two records of data base\n \n",st_df.head(2))

print("\n \n Number of rows and columns",st_df.shape)
print("\n\n")

records = []
for i in range(10):
    records.append([str(st_df.values[i,j]) for j in range(2)])
records[0]

association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)

print("First rule in list \n",association_results[0])
print("\n")


# In[15]:


pip install mlxtend


# In[19]:


import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

dataset=[['Milk','Onion','Nutmeg','Kidney Beans','Eggs','Yogurt'],
         ['Dill','onion','Nutmeg','Kidney Beans','Eggs','Yogurt'],
        ['Milk','Apple','Kidney Beans','Eggs'],
         ['Milk','Unicorn','Corn','Kidney Beans','Yogurt'],
         ['Corn','onion','onion','Kidney Beans','Ice cream','Eggs']]
         
te=TransactionEncoder()
te_ary=te.fit(dataset).transform(dataset)
         
df=pd.DataFrame(te_ary,columns=te.columns_)
print(df.head())
         
tb_df=fpgrowth(df,min_support=0.6,use_colnames=True)
print(tb_df)


# In[ ]:





# In[6]:





# In[ ]:





# In[ ]:





# In[ ]:




