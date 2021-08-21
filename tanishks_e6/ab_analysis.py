#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys


OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value:  {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value:  {more_searches_p:.3g} \n'
    '"Did more/less instructors use the search feature?" p-value:  {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value:  {more_instr_searches_p:.3g}'
)


import pandas as pd
import scipy.stats as s


# In[4]:


def main():
    searchdata_file = sys.argv[1]
    search = pd.read_json(searchdata_file, orient = 'records', lines = True) #odd = new, even = old
    
    even_uid = search[search['uid'] % 2 == 0]
    odd_uid = search[search['uid'] % 2 != 0]
    
    
    even_gzero = even_uid[even_uid['search_count'] > 0]['search_count'].count()
    even_zero = even_uid[even_uid['search_count'] == 0]['search_count'].count()
    
    odd_gzero = odd_uid[odd_uid['search_count'] > 0]['search_count'].count() 
    odd_zero = odd_uid[odd_uid['search_count'] == 0]['search_count'].count()
    
    chi_contingency = [[even_gzero, even_zero], [odd_gzero, odd_zero]]
    chi, p, dof, exp = s.chi2_contingency(chi_contingency) #chi data for all users
    
    iOdd_uid = odd_uid[odd_uid['is_instructor'] == True]
    iEven_uid = even_uid[even_uid['is_instructor'] == True]
    
    iEven_gzero = iEven_uid[iEven_uid['search_count'] > 0]['search_count'].count()
    iEven_zero = iEven_uid[iEven_uid['search_count'] == 0]['search_count'].count()
    
    iOdd_gzero = iOdd_uid[iOdd_uid['search_count'] > 0]['search_count'].count() 
    iOdd_zero = iOdd_uid[iOdd_uid['search_count'] == 0]['search_count'].count()
    
    iChi_contingency = [[iEven_gzero, iEven_zero], [iOdd_gzero, iOdd_zero]]
    iChi, iP, iDof, iExp = s.chi2_contingency(iChi_contingency)
    

    # Output
    print(OUTPUT_TEMPLATE.format(
        more_users_p = p,
        more_searches_p = s.mannwhitneyu(odd_uid['search_count'], even_uid['search_count'], alternative='two-sided').pvalue,
        more_instr_p = iP,
        more_instr_searches_p = s.mannwhitneyu(iOdd_uid['search_count'], iEven_uid['search_count'], alternative='two-sided').pvalue,
    ))


if __name__ == '__main__':
        main()


# In[ ]:




