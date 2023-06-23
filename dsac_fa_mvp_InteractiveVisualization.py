#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import plotly.express as px

# Import the visualization class from dsac_fa_mvp_Visualization.py
from dsac_fa_mvp_Visualization import visualization
from dsac_fa_mvp_Storage import tesla_data
from dsac_fa_mvp_Storage import ford_data

app = dash.Dash(__name__)

# Create an instance of the visualization class
visual = visualization()

app.layout = html.Div([
    html.H1("Stock Data Dashboard"),
    dcc.Dropdown(
        id='plot-dropdown',
        options=[
            {'label': 'Tesla Open Price', 'value': 'open_price'},
            {'label': 'Comparison of Capitalization', 'value': 'capitalism'},
            {'label': 'Return on Investment', 'value': 'roi'},
            {'label': 'Monthly Average Stock Price', 'value': 'monthly_avg'}
        ],
        value='open_price'
    ),
    dcc.Graph(id='plot-graph')
])

@app.callback(
    Output('plot-graph', 'figure'),
    [Input('plot-dropdown', 'value')]
)
def update_plot(selected_plot):
    if selected_plot == 'open_price':
        return visual.open_price(tesla_data)
    elif selected_plot == 'capitalism':
        return visual.capitalism(ford_data, tesla_data)
    elif selected_plot == 'roi':
        return visual.roi(tesla_data)
    elif selected_plot == 'monthly_avg':
        return visual.monthly_avg(ford_data)

if __name__ == '__main__':
    app.run_server()


# In[ ]:




