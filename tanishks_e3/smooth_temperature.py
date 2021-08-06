#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv


# In[2]:


import pandas as pd
from datetime import datetime, timedelta 


# In[3]:


import sys


# In[4]:


path = sys.argv[1]


# In[5]:


tempData = pd.read_csv('C:/Users/Tanis/3D Objects/CMPT 353/Ex3/sysinfo.csv');


# In[6]:


type(tempData['timestamp'])


# In[7]:


#Converting tempData from Pandas Series to datetime
tempData['timestamp'] = pd.to_datetime(tempData['timestamp'])
tempData['datetimestamp'] = tempData['timestamp']


# In[8]:


tempData['timestamp']


# In[9]:


tempData.dtypes


# In[10]:


import numpy as np


# In[11]:


tempData['timestamp']


# In[12]:


tempData['timestamp'] = (tempData['timestamp'] - pd.to_datetime('1970-01-01')) / np.timedelta64(1, 's')


# In[13]:


from statsmodels.nonparametric.smoothers_lowess import lowess


# In[14]:


LOESS = lowess(tempData['temperature'], tempData['timestamp'], frac = 0.01)


# In[15]:


import matplotlib.pyplot as plt


# In[16]:


plt.figure(figsize=(20, 10))
plt.plot(tempData['timestamp'], tempData['temperature'], 'b.', alpha=0.5)
plt.plot(tempData['timestamp'], LOESS[:, 1], 'r-', alpha=1)


# In[17]:


kalman_data = tempData[['temperature', 'cpu_percent', 'sys_load_1', 'fan_rpm']]


# In[18]:


from pykalman import KalmanFilter


# In[19]:


import statistics


# In[20]:


stdTemp = statistics.stdev(tempData['fan_rpm'])**2


# In[21]:


stdTemp


# In[22]:


initial_state = kalman_data.iloc[0]
#observation_covariance = np.diag([0.6, 5, 0.2,  7]) ** 2
observation_covariance = np.diag([10.933409756351889, 49.164693079449755, 0.6509387284828728,  4755.023101973278]) ** 2
transition_covariance = np.diag([0.00001, 0.07, 0.0005, 0.07]) ** 2
transition = [[0.97 , 0.5 , 0.2 , -0.001 ], [0.1 , 0.4 , 2.2  , 1], [0, 0, 0.95, 0], [0,0,0,1]]


# In[23]:



kf = KalmanFilter(initial_state_mean = initial_state, observation_covariance = observation_covariance, transition_covariance = transition_covariance, transition_matrices = transition)
kalman_smoothed = kf.smooth(kalman_data)[0]


plt.figure(figsize =(20,10))

plt.plot(tempData['datetimestamp'], tempData['temperature'], 'b.', alpha=0.5)

plt.plot(tempData['datetimestamp'], LOESS[:, 1], 'r-', alpha=1)

plt.plot(tempData['datetimestamp'], kalman_smoothed[:,0], 'k-', alpha=1)

plt.legend(['CPU Data', 'LOESS Smoothing', 'Kalman Smoothing'])

plt.savefig('cpu.svg')

plt.show()


# In[ ]:





# In[ ]:




