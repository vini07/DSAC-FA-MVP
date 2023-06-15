#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
from dsac_fa_mvp_Storage import tesla_data
from dsac_fa_mvp_Storage import ford_data


# In[18]:


class statistical_analysis():
    
    def statistical_characteristics(self, label, data):
        print("Statistical Characterisitics for", label, ":")
        display(data.describe())
        

analysis = statistical_analysis()
analysis.statistical_characteristics("Tesla", tesla_data)
analysis.statistical_characteristics("Ford", ford_data)


# In[ ]:




