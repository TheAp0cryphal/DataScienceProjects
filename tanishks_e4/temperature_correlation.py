#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


# In[2]:


import math


# In[3]:


def distance(city, stations):
    
    city_lat = city['latitude']
    city_lon = city['longitude']
    station_lat = stations['latitude']
    station_lon = stations['longitude']
    
    p = math.pi/180
    x = 0.5 - np.cos((station_lat-city_lat)*p)/2 + np.cos(city_lat*p) * np.cos(station_lat*p) * (1- np.cos((station_lon-city_lon)*p))/2
    return 12742*np.arcsin(np.sqrt(x))


# In[4]:


def best_tmax(city, stations):
    stations['distance'] = distance(city, stations)
    closest = stations['distance'].argmin()
    avg_tmax = stations.loc[closest]["avg_tmax"]
    return avg_tmax
    


# In[5]:


def main():
    
    stations_file = sys.argv[1]
    city_file = sys.argv[2]
    output = sys.argv[3]

    stations = pd.read_json(stations_file, lines=True)
    stations['avg_tmax'] = stations['avg_tmax'] / 10.0

    cities = pd.read_csv(city_file)
    cities = cities.dropna()
    cities['area'] = cities['area'].astype(float)
    cities['area'] = cities['area'] / 1000000.0
    cities = cities[cities['area'] <= 10000.0]  
    cities = cities.reset_index(drop = True)
         
    cities['avg_tmax'] = cities.apply(best_tmax, axis = 1, stations = stations)  
    cities['density'] = cities['population'] / cities['area']
    
    plt.ticklabel_format(style='plain', axis='both')
    plt.scatter(cities['avg_tmax'], cities['density'])
   
    plt.xlabel = "Avg Max Temperature (\u00b0C)"
    plt.ylabel = "Population Density (people/km\u00b2)"
    plt.savefig(output)
    plt.show()


# In[6]:



if __name__ == '__main__':
    main()
    


# In[ ]:




