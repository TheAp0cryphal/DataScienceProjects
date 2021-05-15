#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
totals = pd.read_csv ('totals.csv').set_index(keys = ['name'])
counts = pd.read_csv ('counts.csv').set_index(keys = ['name'])

print("City with lowest total precipitation:\n")
rowSum = totals.sum(axis = 1)

print (rowSum.idxmin())


totalCount = counts.sum(axis=0)


print ("\n Average Precipitation in each month:\n")


print (colSum/totalCount)

countRowSum = counts.sum(axis = 1)
print("\n Average Precipitation in each city:\n")

print (rowSum/countRowSum)


# In[ ]:




