#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
dia_data = pd.read_csv(r"C:\Users\ISE2019\Downloads\dm dataset\dia-small.csv")
#Display the first 10 rows
records = dia_data.head(10)
print("First 10 rows of the records:",records)


# In[18]:


import pandas as pd 
import numpy as np
a1 = pd.Series ([2,3,4]) 
a1.describe ()


# In[23]:


import pandas as pd 
import numpy as np
dia_data = pd.read_csv(r"C:\Users\ISE2019\Downloads\dm dataset\data1.csv")
print (dia_data.shape) 
dia_data.dropna (inplace = True)
print (dia_data.shape)


# In[22]:


import pandas as pd 
import numpy as np
dia_data = pd.read_csv(r"C:\Users\ISE2019\Downloads\dm dataset\dia-small-only10.csv")
print("before use of drop na()",dia_data.shape)
dia_data.dropna(axis=1,how='all',inplace=True)
print("After use of drop na()",dia_data.shape)
dia_data.head(10)


# In[ ]:




