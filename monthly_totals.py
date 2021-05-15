#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


def date_to_month(d):
    # You may need to modify this function, depending on your data types.
    return '%02i' % (d.month)

def get_precip_data():
    return pd.read_csv('precipitation.csv', parse_dates=[2])

def pivot_months_pandas(data):
    """
    Create monthly precipitation totals for each station in the data set.
    
    This should use Pandas methods to manipulate the data.
    """
    
    
    df = data[['name','date','precipitation']]
    
    df.date = df.date.apply(date_to_month)
    
    
    monthly = df.groupby(['name', 'date']).agg('sum')
    counts = df.groupby(['name','date']).agg('count')
    
    monthly = monthly.pivot_table(index = 'name', columns = 'date', values = 'precipitation')
    counts = counts.pivot_table(index = 'name', columns = 'date', values = 'precipitation')
    
    return monthly, counts


def main():
    data = get_precip_data()
    totals, counts = pivot_months_pandas(data)
    totals.to_csv('totals.csv')
    counts.to_csv('counts.csv')
    np.savez('monthdata.npz', totals=totals.values, counts=counts.values)

if __name__ == '__main__':
    main()


# In[ ]:




