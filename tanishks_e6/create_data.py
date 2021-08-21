#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd

import time
from implementations import all_implementations

sub = np.zeros((150, 7))

for i in range(150):
    j = 0
    random_array = np.random.randint(-1000, 1000, 2000)
    for sort in all_implementations:
        st = time.time()
        res = sort(random_array)
        en = time.time()
        sub[i][j] = en - st
       
        if (j < 7):
            j = j + 1

df = pd.DataFrame(list(sub), columns = ['qs1','qs2', 'qs3', 'qs4', 'qs5', 'merge1', 'partition_sort'])
df.to_csv('data.csv', index = False)


# In[ ]:




