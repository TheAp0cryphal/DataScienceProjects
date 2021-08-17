#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sys


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


OUTPUT_TEMPLATE = (    
"Initial T-test p-value: {initial_ttest_p:.3g}\n"
"Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
"Original data equal-variance p-value: {initial_levene_p:.3g}\n"
"Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
"Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
"Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
"Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
"Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
"Mann-Whitney U-test p-value: {utest_p:.3g}"    
)


# In[3]:



file = sys.argv[1]

counts = pd.read_json(file, lines=True)

counts['date'] = pd.to_datetime(counts['date'])

mask = (counts['date'] > '2011-12-31') & (counts['date'] < '2014-1-1')

counts = counts.loc[mask]

#https://stackoverflow.com/questions/29370057/select-dataframe-rows-between-two-dates

counts = counts[counts['subreddit'] == 'canada'] 

counts = counts.reset_index(drop = True)

counts


# In[ ]:


import datetime as dt


# In[ ]:


weekend = (counts[counts['date'].dt.dayofweek >= 5])

weekday = (counts[counts['date'].dt.dayofweek < 5])


# In[ ]:


weekday['yearnweek'] = weekday['date'].dt.date


# In[ ]:


#https://stackoverflow.com/questions/48058304/how-to-apply-series-in-isocalendar-function-in-pandas-python
weekday["yearnweek"] = weekday['yearnweek'].apply(lambda x: str(x.isocalendar()[0]) + '-' + str(x.isocalendar()[1]).zfill(2))
mWeekday = weekday.groupby(['yearnweek']).aggregate('mean')['comment_count']


# In[ ]:


weekend["yearnweek"] = weekend['date'].dt.date
weekend["yearnweek"] = weekend['yearnweek'].apply(lambda x: str(x.isocalendar()[0]) + '-' + str(x.isocalendar()[1]).zfill(2))
mWeekend = weekend.groupby(['yearnweek']).aggregate('mean')['comment_count']


# In[ ]:


import scipy.stats as stats

import matplotlib.pyplot as plt


# In[ ]:


plt.hist(weekend['comment_count'])


# In[ ]:



print(OUTPUT_TEMPLATE.format(
        
        initial_ttest_p = stats.ttest_ind (
            weekday['comment_count'], weekend['comment_count']).pvalue ,
        
        initial_weekday_normality_p = stats.normaltest(
            weekday['comment_count']).pvalue ,
        
        initial_weekend_normality_p = stats.normaltest(
            weekend['comment_count']).pvalue ,
        
        initial_levene_p = stats.levene(
            weekday['comment_count'], weekend['comment_count']).pvalue ,
        
        transformed_weekday_normality_p = stats.normaltest(
            np.sqrt(weekday['comment_count'])).pvalue ,
            #np.log(weekday['comment_count'])).pvalue
            #np.exp(weekday['comment_count'])).pvalue
            #(weekday['comment_count']**2)).pvalue
        
        transformed_weekend_normality_p = stats.normaltest(
            np.sqrt(weekend['comment_count'])).pvalue ,
            #np.log(weekend['comment_count'])).pvalue,
            #np.exp(weekend['comment_count'])).pvalue,
            #(weekend['comment_count']**2)).pvalue,
        
        transformed_levene_p = stats.levene(
            np.sqrt(weekday['comment_count']), np.sqrt (weekend['comment_count'])).pvalue ,
            #np.log(weekday['comment_count']), np.log (weekend['comment_count'])).pvalue,
            #np.exp(weekday['comment_count']), np.log (weekend['comment_count'])).pvalue,
            #(weekday['comment_count']**2), np.log (weekend['comment_count']**2)).pvalue,
        
        weekly_weekday_normality_p = stats.normaltest(
            mWeekday).pvalue ,
        
        weekly_weekend_normality_p = stats.normaltest(
            mWeekend).pvalue ,
        
        weekly_levene_p = stats.levene(
            mWeekday, mWeekend).pvalue ,
        
        weekly_ttest_p = stats.ttest_ind(
            mWeekday, mWeekend).pvalue ,
        
        utest_p = stats.mannwhitneyu(
            weekday['comment_count'], weekend['comment_count'], alternative = 'two-sided').pvalue
    ))


# In[ ]:




