#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import difflib
import numpy as np
import sys


# In[2]:


movie_list = sys.argv[1]
movie_ratings = sys.argv[2]
output = sys.argv[3]


# In[3]:


movies = open(movie_list).read().splitlines()


# In[4]:


moviesDF = pd.DataFrame(movies, columns=["title"])


# In[5]:


movie_ratings =  pd.read_csv(movie_ratings)


# In[6]:


def fmatch(word):
    word_match = difflib.get_close_matches(word, moviesDF['title'], cutoff = 0.7)
    return word_match
    


# In[7]:


print(fmatch('Dog Day yfternoon'))


# In[8]:


movie_ratings['title'] = movie_ratings['title'].apply(lambda x: fmatch(x))


# In[9]:


movie_ratings['title']


# In[10]:


def listToString(x):
    string = ""
    return (string.join(x))


# In[11]:


movie_ratings['title'] = movie_ratings['title'].apply(lambda x: listToString(x))


# In[12]:


movie_ratings = movie_ratings[movie_ratings.title != '']


# In[13]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# In[14]:


movie_ratings['rating'] = movie_ratings['rating'].astype(int)


# In[16]:


movie_ratings.groupby('title')["rating"].mean().round(2).reset_index().to_csv(output, index=False)


# In[ ]:




