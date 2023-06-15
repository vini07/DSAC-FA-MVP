#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from dsac_fa_mvp_Storage import tesla_data
from dsac_fa_mvp_Storage import ford_data


# In[38]:


class visualization():
    def open_price(self, data):
        reset_data = data.reset_index()
        fig = px.line(reset_data, x="Date", y="Open", title='Stock Open Prices For Tesla')
        fig.show()
        
    def capitalism(self, data1, data2):
        # Market Capitalisation = 
        reset_data1 = data1.reset_index()
        reset_data2 = data2.reset_index()
        
        # Calculate market capitalization for each dataset
        reset_data1['Ford'] = reset_data1['Close'] * reset_data1['Volume']
        reset_data2['Tesla'] = reset_data2['Close'] * reset_data2['Volume']
        
        # Merge the two datasets based on the 'Date' column
        merged_data = pd.merge(reset_data1, reset_data2, on='Date', how='outer')
        
        fig = px.line(merged_data, x="Date", y=["Ford", "Tesla"], title='Comparison of Capitalization for Ford and Tesla')
        fig.show()
        

    def roi(self, data):
        # Return on Investment
        reset_data = data.reset_index()
        reset_data.set_index('Date', inplace=True)

        # Resample data to biweekly frequency and calculate ROI
        biweekly_data = reset_data.resample('2W').first()  # Resample to biweekly frequency using the first day of each period
        biweekly_data['Open'] = reset_data['Open'].resample('2W').first()  # Use the opening price on the first day 
        biweekly_data['Close'] = reset_data['Close'].resample('2W').last()  # Use the closing price on the last day 
        biweekly_data['roi'] = ((biweekly_data['Close'] - biweekly_data['Open']) / biweekly_data['Open']) * 100

        fig = px.line(biweekly_data.reset_index(), x="Date", y="roi", title='Biweekly Return on Investment for Tesla')
        fig.show()
        
    def monthly_avg(self, data):
        # Monthly Average of Opening stock price
        reset_data = data.reset_index()
        reset_data['avg'] = reset_data['Open'].rolling(30).mean() # Count average of opening price for 30 Days
        fig = px.line(reset_data, x="Date", y="avg", title='Monthly Average Stock Price for Ford')
        fig.add_scatter(x=reset_data['Date'], y=reset_data['Open'], name="Actual Price")
        fig.show()
        
        
        
visual = visualization()
visual.open_price(tesla_data)
visual.capitalism(ford_data, tesla_data)
visual.roi(tesla_data)
visual.monthly_avg(ford_data)


# In[ ]:




