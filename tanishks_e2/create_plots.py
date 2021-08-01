#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import sys 


# In[ ]:


filenameA = sys.argv[1]
filenameB = sys.argv[2]


# In[ ]:


#Pareto Distribution

data =  pd.read_csv(filenameA, sep=' ', header=None, index_col=1,
        names=['lang', 'page', 'views', 'bytes'])


# In[ ]:


data = data.sort_values(by='views', ascending = False)


# In[ ]:


print(data)


# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(20,10)) # change the size to something sensible
plt.subplot(1, 2, 1) # subplots in 1 row, 2 columns, select the first
plt.plot(data.views.values) # build plot 1
plt.ylabel('Page Views')


# In[ ]:


data2 =  pd.read_csv(filenameB, sep=' ', header=None, index_col=1,
        names=['lang', 'page', 'views', 'bytes'])


# In[ ]:


data1 =  pd.read_csv(filenameA, sep=' ', header=None, index_col=1,
        names=['lang', 'page', 'views', 'bytes'])


# In[ ]:


data2.insert(3, "views2", '1')


# In[ ]:


data2['views2'] = data1['views']


# In[ ]:


print(data2)


# In[4]:



plt.figure(figsize=(12,6))
plt.subplot(1,2,2)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Hour 1')
plt.xlabel('Hour 2')
plt.title('Hourly Comparison of Views')
plt.scatter(data2['views'], data2['views2'])
plt.show()
savefile = plt.gcf()
savefile.savefig('wikipedia.png')


# In[ ]:




