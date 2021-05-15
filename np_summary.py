#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np

data = np.load('monthdata.npz')
totals = data['totals']
counts = data['counts']

rowSum = np.sum(totals, axis = 1)

colSum = np.sum(totals, axis = 0)

totalCount = np.sum(counts, axis=0)

print("Row with lowest total precipitation:\n")

print (np.argmin(rowSum))

print ("\n Average Precipitation in each month\n")

print (colSum/totalCount)

print("\n Average Precipitation in each city:\n")

countRowSum = np.sum(counts, axis = 1)

print (rowSum/countRowSum) # Average Precipation in each city


reshaped = np.reshape(totals, (36,3))

quarterSum = np.sum(reshaped, axis = 1)


quarterSum = np.reshape(quarterSum, (9,4))

print("\nQuaterly Precipitation totals:\n")

print (quarterSum)



# In[ ]:




