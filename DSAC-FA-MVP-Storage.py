#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install firebase-admin


# In[2]:


import yfinance as yf
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Initialize Firebase app
cred = credentials.Certificate('C:/Users/PAPU/Pictures/lambton-wil-project1-firebase-adminsdk-az9ud-854033ce28.json')
firebase_admin.initialize_app(cred)
db = firestore.client()


# In[4]:


class StockData:

    start_date = '2021-01-01' 
    end_date = '2021-12-31'
    def fetch_data(self, ticker):
        data = yf.download(ticker, start=self.start_date, end=self.end_date)

        # Extract necessary data fields
        stock_data = {
            'Open': data['Open'].tolist(),
            'High': data['High'].tolist(),
            'Low': data['Low'].tolist(),
            'Close': data['Close'].tolist(),
            'Volume': data['Volume'].tolist(),
            'Adj Close': data['Adj Close'].tolist()
        }

        # Store data to Firebase
        doc_ref = db.collection('stock_data').document(ticker)
        doc_ref.set(stock_data)

        return data


# In[8]:


# Create instance of StockData class
stock = StockData()

# Collect stock data for Tesla
tesla_data = stock.fetch_data('TSLA')
print("Stock Data of Tesla")
print(tesla_data)

# Collect stock data for Ford
ford_data = stock.fetch_data('F')
print("Stock Data of Ford")
print(ford_data)

