#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd

from math import cos, asin, sqrt, pi

import numpy as np

from pykalman import KalmanFilter

import xml.etree.ElementTree as ET


# In[2]:


def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')


# In[3]:



def parse_data(gpx):
    
    parse_result = ET.parse(gpx)
    GPX = pd.DataFrame(columns = ['lat', 'lon'])
    for i in parse_result.iter('{http://www.topografix.com/GPX/1/0}trkpt'):
        series = pd.Series([i.get('lat'), i.get('lon')], index = ['lat','lon'])
        GPX = GPX.append(series, ignore_index = True)
    return GPX
#https://stackoverflow.com/questions/28259301/how-to-convert-an-xml-file-to-nice-pandas-dataframe


# In[4]:


def distance(lat1, lon1, lat2, lon2):
    lat1 = float(lat1)
    lon1 = float(lon1)
    lat2 = float(lat2)
    lon2 = float(lon2)
    p = pi/180
    x = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1- cos((lon2-lon1)*p))/2
    return 12742*asin(sqrt(x))

#https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206


# In[ ]:





# In[5]:


def smooth(gpx):
    print(type(gpx))
    initial_state = gpx.iloc[0]
    observation_covariance = np.diag([0.25, 0.25]) ** 2
    transition_covariance = np.diag([0.1 ,0.1]) ** 2
    transition = ([1,0],[0,1])
    gpxLat= pd.to_numeric(gpx['lat'])
    gpxLon= pd.to_numeric(gpx['lon'])
    frames = [gpxLat, gpxLon]
    gpx = pd.concat(frames, axis=1, join = 'inner')

    
    kf = KalmanFilter(initial_state_mean = initial_state, observation_covariance = observation_covariance, transition_covariance = transition_covariance, transition_matrices = transition)
    kalman_smoothed, _ = kf.smooth(gpx)
    return kalman_smoothed
    
    


# In[6]:


def main():
    gpx1 = parse_data(sys.argv[1])
    gpx2 = gpx1.shift(-1);
    frames = [gpx1, gpx2]
    DF = pd.concat(frames, axis=1, join = 'inner')
    
    dist = 0
    
    for i in range(0, len(DF)-1):        
        dist = dist + (distance(DF.iat[i,0],
                                DF.iat[i,1],
                                DF.iat[i,2],
                                DF.iat[i,3]))
    dist = dist * 1000
    print ('Unfiltered Distance: %0.2f' % round(dist,2))
    smoothed_points1 = smooth(gpx1)
    smoothed_points1 = pd.DataFrame(smoothed_points1, columns = ['lat','lon'])
    smoothed_points2 = smoothed_points1.shift(-1)
    frames = [smoothed_points1, smoothed_points2]
    smoothDF = pd.concat(frames, axis = 1, join = 'inner')
    smoothdist = 0
    for i in range(0, len(DF)-1):        
        smoothdist = smoothdist + (distance(smoothDF.iat[i,0],
                                smoothDF.iat[i,1],
                                smoothDF.iat[i,2],
                                smoothDF.iat[i,3]))
    
    smoothdist = smoothdist * 1000
    print ('Filtered Distance: %0.2f' % round(smoothdist,2))
    print(smoothed_points1)
    output_gpx(smoothed_points1, 'out.gpx')


# In[7]:


if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




