#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# In[3]:


OUTPUT_TEMPLATE = (
    'Score = {score:.3f}\n'
)


# In[4]:


def main():
    trainingData = sys.argv[1]
    predictionData = sys.argv[2]
    trainingData = pd.read_csv(trainingData)
    predictionData = pd.read_csv(predictionData)
    y = trainingData['city']
    X = trainingData.drop(['city'], axis = 1) # dropping city data to create prediciton relationship between X and y 
    predict = predictionData.drop(['city'], axis = 1) 
    
    X_train, X_valid, y_train, y_valid = train_test_split(X , y)
    
    rcf = make_pipeline(
         StandardScaler(),# normalizing the data
         RandomForestClassifier(n_estimators = 300, max_depth = 15, min_samples_leaf = 7)
    )
    
    rcf.fit(X_train, y_train)
    
    predictions = rcf.predict(predict)
    pd.Series(predictions).to_csv(sys.argv[3], index = False, header = False)
    
    print(OUTPUT_TEMPLATE.format(
        score = rcf.score(X_valid, y_valid)
    ))
    


# In[5]:


if __name__ == '__main__':
    main()


# In[ ]:




