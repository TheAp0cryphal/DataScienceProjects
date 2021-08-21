#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as s
import sys

file = sys.argv[1]
df = pd.read_csv(file)


onewayanova = s.f_oneway(df['qs1'],df['qs2'], df['qs3'], df['qs4'], df['qs5'], df['merge1'], df['partition_sort'])

melt = pd.melt(df)
posthoc = pairwise_tukeyhsd(melt['value'], melt['variable'], alpha = 0.05)
print(posthoc)
plot = posthoc.plot_simultaneous()


# In[ ]:




