#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the required modules 

import os 
import numpy as np
import pandas as pd 
import math 
import datetime as dt 
import matplotlib.pyplot as plt 


# In[2]:


# using scikit learn for model evaluation

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler 


# In[3]:


# using tensorflow for buidling the model 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM


# In[4]:


# using these libraries for plotting

import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# In[5]:


maindf = pd.read_csv('btc_1h.csv')
maindf = pd.read_csv('btc_2h.csv')
maindf = pd.read_csv('btc_3m.csv')
maindf = pd.read_csv('btc_4h.csv')
maindf = pd.read_csv('btc_5m.csv')
maindf = pd.read_csv('btc_6h.csv')
maindf = pd.read_csv('btc_15m.csv')
maindf = pd.read_csv('btc_30m.csv')


# In[ ]:





# In[6]:


print("total number of the days present in the datset are: ", maindf.shape[0])
print("total number of fields present in the dataset are: ", maindf.shape[1])


# In[7]:


maindf.shape


# 
# # high - highest price  of the bitcoin 
# 
# # low - lowest price of the bitcoin
# 
# # close - closing price of the bitcoin 

# In[8]:


maindf.describe()


# In[9]:


maindf.info()


# In[10]:


maindf.head() 


# In[11]:


maindf.tail() 


# In[12]:


print("The null values are: ", maindf.isnull().values.sum())
print("NA Values? : ", maindf.isnull().values.any()) # should return false if there are no NULL Values. 


# In[13]:


"""
# if null values are to be found later, use the following code to remove them!

maindf = maindf.dropna()
print("Null values: ", maindf.isnull().values.sum())
print("NA values: ", maindf.isnull().values.any())

"""


# # Lets perform the Exploratory Data Analysis 
# 
# iloc (integer-location based indexing 
# for the start date - It is equivalent to saying "get the value in the cell at the first row and first column."
# for the end date - it is used to get the value of the last row and the last column 
# 
# (-1 is used for indicating the last row in the dataset) 
# (0 indicates the first column of the last row, when we select the last row as -1, we have to select the first col, as 0) 
# 
# 

# In[14]:


start_date = maindf.iloc[0][0]
end_date = maindf.iloc[-1][0]

print("Starting date is: ", start_date)
print("Ending date is: ", end_date)


# # Analysis of the year 2018 

# In[15]:


maindf['datetime'] = pd.to_datetime(maindf['datetime'], format='%Y-%m-%d %H:%M:%S')  # Adjust the format as needed

y_2018 = maindf.loc[(maindf['datetime'] >= '2018-01-01') & (maindf['datetime'] < '2019-01-01')]

# Drop specified columns and assign the result back to y_2018
y_2018 = y_2018.drop(['volume'], axis=1)

"""

the code prepares the data for analysis by converting the date column to a datetime format,
Filtering the data to include only the year 2018
removing the 'Adj Close' and 'Volume' columns from the filtered data. 
The final result is stored in the DataFrame y_2018.

"""


# In[16]:


import pandas as pd

# Define the list of dataset file names
dataset_files = ["btc_1h.csv", "btc_2h.csv", "btc_3m.csv", "btc_4h.csv", "btc_5m.csv", "btc_6h.csv", "btc_15m.csv", "btc_30m.csv"]

# Loop through each dataset file
for file in dataset_files:
    # Load the data
    dataset = pd.read_csv(file)  
    
    # Convert 'datetime' column to datetime format
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    
    # Filter data for the year 2018
    dataset_2018 = dataset[dataset['datetime'].dt.year == 2018]
    
    # Set 'datetime' column as the index
    dataset_2018.set_index('datetime', inplace=True)

    # Extract the time interval from the file name
    time_interval = file.split('_')[-1].split('.')[0]

    # Print the last five rows for each dataset in 2018
    print(f"Last five rows for {time_interval} in 2018:")
    print(dataset_2018.tail())
    print("\n" + "="*50 + "\n")  # Separate results for each dataset


# In[17]:


import pandas as pd

def preprocess_dataset(dataset, time_interval='1H', year=None):
    # Add a condition to filter data for a specific year
    if year is not None:
        dataset = dataset[dataset.index.year == year]

    # Assuming 'Close' column is the closing prices in the dataset
    dataset['open'] = dataset['close'].shift(1)

    # Fill NaN in the first row with the initial closing price (you may choose another strategy)
    dataset['open'].fillna(dataset['close'][0], inplace=True)

    # Filter data based on the provided time interval
    dataset_filtered = dataset.resample(time_interval).last()

    # Drop NaN rows that might occur due to resampling
    dataset_filtered = dataset_filtered.dropna()

    # Group by month and calculate mean
    monthwise = dataset_filtered.groupby(dataset_filtered.index.month)[['open', 'close']].mean()

    # Ensure all months are present in the results
    all_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    monthwise = monthwise.reindex(all_months)

    # Rename the index to month names
    monthwise.index = monthwise.index.map({1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                                           7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'})

    # Add year information to the results
    monthwise['Year'] = year

    return monthwise

# Define the list of dataset file names
dataset_files = ["btc_1h.csv", "btc_2h.csv", "btc_3m.csv", "btc_4h.csv", "btc_5m.csv", "btc_6h.csv", "btc_15m.csv", "btc_30m.csv"]

# Loop through each dataset file
for file in dataset_files:
    # Load the data
    dataset = pd.read_csv(file)  
    
    # Set 'datetime' column as the index
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    dataset.set_index('datetime', inplace=True)

    # Extract the time interval from the file name
    time_interval = file.split('_')[-1].split('.')[0]

    # Preprocess the data for the year 2018
    result = preprocess_dataset(dataset, time_interval=time_interval, year=2018)

    # Print the result
    print(f"Results for {time_interval} in 2018:")
    print(result)
    print("\n" + "="*50 + "\n")  # Separate results for each dataset


# # 1) we initialize a blank figure for the plot first using the fig = go.figure. Next, we have to add the traces to the figure, two for opening and two for closing. 
# 
# # 2) monthwise index for x sets the x axis values to the months, while the monthwise for y axis sets avg opening and closing values. 
# 
# # 3) we rotate the x axis labels for better readability. (-45 is commonly used angle for rotating text on plots, as its a standard angle for wide range of scenarios!)

# In[18]:


import plotly.graph_objects as go
import pandas as pd

# Define the order of months
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# List of dataset file names
dataset_files = ["btc_1h.csv", "btc_2h.csv", "btc_3m.csv", "btc_4h.csv", "btc_5m.csv", "btc_6h.csv", "btc_15m.csv", "btc_30m.csv"]

# Define colors for each dataset
colors = ['rgb(0, 128, 0)', 'rgb(255, 165, 0)', 'rgb(0, 0, 128)', 'rgb(128, 0, 0)', 'rgb(128, 0, 128)', 'rgb(0, 128, 128)', 'rgb(255, 0, 0)', 'rgb(0, 255, 0)']

# Create an empty figure
fig = go.Figure()

# Loop through each dataset file
for file, color in zip(dataset_files, colors):
    # Load the data
    dataset = pd.read_csv(file)
    
    # Convert 'datetime' column to datetime format
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    
    # Filter data for the year 2018
    dataset_2018 = dataset[dataset['datetime'].dt.year == 2018]

    # Calculate monthwise mean closing prices
    monthwise_mean = dataset_2018.groupby(dataset_2018['datetime'].dt.strftime('%B'))['close'].mean()
    monthwise_mean = monthwise_mean.reindex(new_order, axis=0)

    # Add traces to the figure with different colors
    fig.add_trace(go.Bar(
        x=monthwise_mean.index,
        y=monthwise_mean,
        name=f'{file} Mean Closing Price',
        marker_color=color
    ))

# Update layout and show the figure
fig.update_layout(barmode="group",
                  title='Monthwise Mean Closing Price of Bitcoin (All Datasets) - 2018',
                  xaxis_title='Month',
                  yaxis_title='Mean Closing Price')

fig.show()

# mean is being calculated using the closing price for the bar graph


# In[19]:


# comparison between the mean and the original closing price 
import plotly.graph_objects as go
import pandas as pd

# Define the order of months
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# List of dataset file names
dataset_files = ["btc_1h.csv", "btc_2h.csv", "btc_3m.csv", "btc_4h.csv", "btc_5m.csv", "btc_6h.csv", "btc_15m.csv", "btc_30m.csv"]

# Define colors for each dataset
colors = ['rgb(0, 128, 0)', 'rgb(255, 165, 0)', 'rgb(0, 0, 128)', 'rgb(128, 0, 0)', 'rgb(128, 0, 128)', 'rgb(0, 128, 128)', 'rgb(255, 0, 0)', 'rgb(0, 255, 0)']

# Create an empty figure
fig = go.Figure()

# Loop through each dataset file
for file, color in zip(dataset_files, colors):
    # Load the data
    dataset = pd.read_csv(file)
    
    # Convert 'datetime' column to datetime format
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    
    # Filter data for the year 2020
    dataset_2018 = dataset[dataset['datetime'].dt.year == 2018]

    # Calculate monthwise mean closing prices
    monthwise_mean = dataset_2018.groupby(dataset_2018['datetime'].dt.strftime('%B'))['close'].mean()
    monthwise_mean = monthwise_mean.reindex(new_order, axis=0)

    # Calculate monthwise original closing prices
    monthwise_original = dataset_2018.groupby(dataset_2018['datetime'].dt.strftime('%B'))['close'].last()
    monthwise_original = monthwise_original.reindex(new_order, axis=0)

    # Add traces to the figure with different colors for mean and original
    fig.add_trace(go.Bar(
        x=monthwise_mean.index,
        y=monthwise_mean,
        name=f'{file} Mean Closing Price',
        marker_color=color,
        opacity=0.7  # Adjust opacity for better visibility of both bars
    ))

    fig.add_trace(go.Bar(
        x=monthwise_original.index,
        y=monthwise_original,
        name=f'{file} Original Closing Price',
        marker_color=color,
        opacity=0.5  # Adjust opacity for better visibility of both bars
    ))

# Update layout and show the figure
fig.update_layout(barmode="group",
                  title='Monthwise Closing Prices of Bitcoin (Mean and Original) - 2018',
                  xaxis_title='Month',
                  yaxis_title='Closing Price')

fig.show()


# In[20]:


import plotly.express as px
from itertools import cycle

# Assuming 'datetime' is already in datetime format
maindf['open'] = maindf['close'].shift(1)
maindf['open'].fillna(maindf['close'][0], inplace=True)

# Filter data for a specific year (e.g., 2018)
year_filter = (maindf['datetime'] >= '2018-01-01') & (maindf['datetime'] < '2019-01-01')
filtered_data = maindf.loc[year_filter]

# Display 'volume' in the printed output for the filtered data
print(filtered_data[['datetime', 'open', 'high', 'low', 'close', 'volume']].head())

# Plotting using Plotly Express for the filtered data
names = cycle(['Bitcoin Open Price', 'Bitcoin Close Price', 'Bitcoin High Price', 'Bitcoin Low Price', 'Bitcoin Volume'])

fig = px.line(filtered_data, x=filtered_data['datetime'], y=[filtered_data['open'], filtered_data['close'], filtered_data['high'], filtered_data['low'], filtered_data['volume']],
              labels={'datetime': 'datetime', 'value': 'Stock Value'})

fig.update_layout(title_text='Bitcoin Analysis Chart (2018)', font_size=15, font_color='black',
                  legend_title_text='Bitcoin Parameters')

fig.for_each_trace(lambda t: t.update(name=next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

fig.show()


# # Analysis of 2019 

# In[21]:


maindf['datetime'] = pd.to_datetime(maindf['datetime'], format='%Y-%m-%d %H:%M:%S')  # Adjust the format as needed

y_2019 = maindf.loc[(maindf['datetime'] >= '2019-01-01') & (maindf['datetime'] < '2020-01-01')]

# Drop specified columns and assign the result back to y_2019
y_2019 = y_2019.drop(['volume'], axis=1)

"""

the code prepares the data for analysis by converting the date column to a datetime format,
Filtering the data to include only the year 2019
removing the 'Adj Close' and 'Volume' columns from the filtered data. 
The final result is stored in the DataFrame y_2019.

"""


# In[22]:


import pandas as pd

# Define the list of dataset file names
dataset_files = ["btc_1h.csv", "btc_2h.csv", "btc_3m.csv", "btc_4h.csv", "btc_5m.csv", "btc_6h.csv", "btc_15m.csv", "btc_30m.csv"]

# Loop through each dataset file
for file in dataset_files:
    # Load the data
    dataset = pd.read_csv(file)  
    
    # Convert 'datetime' column to datetime format
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    
    # Filter data for the year 2018
    dataset_2019 = dataset[dataset['datetime'].dt.year == 2019]
    
    # Set 'datetime' column as the index
    dataset_2019.set_index('datetime', inplace=True)

    # Extract the time interval from the file name
    time_interval = file.split('_')[-1].split('.')[0]

    # Print the last five rows for each dataset in 2018
    print(f"Last five rows for {time_interval} in 2019:")
    print(dataset_2019.tail())
    print("\n" + "="*50 + "\n")  # Separate results for each dataset


# In[23]:


import pandas as pd

def preprocess_dataset(dataset, time_interval='1H', year=None):
    # Add a condition to filter data for a specific year
    if year is not None:
        dataset = dataset[dataset.index.year == year]
    
    dataset['open'] = dataset['close'].shift(1)

    # Fill NaN in the first row with the initial closing price (you may choose another strategy)
    dataset['open'].fillna(dataset['close'][0], inplace=True)

    # Filter data based on the provided time interval
    dataset_filtered = dataset.resample(time_interval).last()

    # Drop NaN rows that might occur due to resampling
    dataset_filtered = dataset_filtered.dropna()

    # Group by month and calculate mean
    monthwise = dataset_filtered.groupby(dataset_filtered.index.month)[['open', 'close']].mean()

    # Ensure all months are present in the results
    all_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    monthwise = monthwise.reindex(all_months)

    # Rename the index to month names
    monthwise.index = monthwise.index.map({1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                                           7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'})

    # Add year information to the results
    monthwise['Year'] = year

    return monthwise

# Define the list of dataset file names
dataset_files = ["btc_1h.csv", "btc_2h.csv", "btc_3m.csv", "btc_4h.csv", "btc_5m.csv", "btc_6h.csv", "btc_15m.csv", "btc_30m.csv"]

# Loop through each dataset file
for file in dataset_files:
    # Load the data
    dataset = pd.read_csv(file)  
    
    # Set 'datetime' column as the index
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    dataset.set_index('datetime', inplace=True)

    # Extract the time interval from the file name
    time_interval = file.split('_')[-1].split('.')[0]

    # Preprocess the data for the year 2018
    result = preprocess_dataset(dataset, time_interval=time_interval, year=2019)

    # Print the result
    print(f"Results for {time_interval} in 2019:")
    print(result)
    print("\n" + "="*50 + "\n")  # Separate results for each dataset


# In[24]:


import plotly.graph_objects as go
import pandas as pd

# Define the order of months
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# List of dataset file names
dataset_files = ["btc_1h.csv", "btc_2h.csv", "btc_3m.csv", "btc_4h.csv", "btc_5m.csv", "btc_6h.csv", "btc_15m.csv", "btc_30m.csv"]

# Define colors for each dataset
colors = ['rgb(0, 128, 0)', 'rgb(255, 165, 0)', 'rgb(0, 0, 128)', 'rgb(128, 0, 0)', 'rgb(128, 0, 128)', 'rgb(0, 128, 128)', 'rgb(255, 0, 0)', 'rgb(0, 255, 0)']

# Create an empty figure
fig = go.Figure()

# Loop through each dataset file
for file, color in zip(dataset_files, colors):
    # Load the data
    dataset = pd.read_csv(file)
    
    # Convert 'datetime' column to datetime format
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    
    # Filter data for the year 2018
    dataset_2019 = dataset[dataset['datetime'].dt.year == 2019]

    # Calculate monthwise mean closing prices
    monthwise_mean = dataset_2019.groupby(dataset_2019['datetime'].dt.strftime('%B'))['close'].mean()
    monthwise_mean = monthwise_mean.reindex(new_order, axis=0)

    # Add traces to the figure with different colors
    fig.add_trace(go.Bar(
        x=monthwise_mean.index,
        y=monthwise_mean,
        name=f'{file} Mean Closing Price',
        marker_color=color
    ))

# Update layout and show the figure
fig.update_layout(barmode="group",
                  title='Monthwise Mean Closing Price of Bitcoin (All Datasets) - 2019',
                  xaxis_title='Month',
                  yaxis_title='Mean Closing Price')

fig.show()

# mean is being calculated using the closing price for the bar graph


# In[25]:


# comparison between the mean and the original closing price 
import plotly.graph_objects as go
import pandas as pd

# Define the order of months
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# List of dataset file names
dataset_files = ["btc_1h.csv", "btc_2h.csv", "btc_3m.csv", "btc_4h.csv", "btc_5m.csv", "btc_6h.csv", "btc_15m.csv", "btc_30m.csv"]

# Define colors for each dataset
colors = ['rgb(0, 128, 0)', 'rgb(255, 165, 0)', 'rgb(0, 0, 128)', 'rgb(128, 0, 0)', 'rgb(128, 0, 128)', 'rgb(0, 128, 128)', 'rgb(255, 0, 0)', 'rgb(0, 255, 0)']

# Create an empty figure
fig = go.Figure()

# Loop through each dataset file
for file, color in zip(dataset_files, colors):
    # Load the data
    dataset = pd.read_csv(file)
    
    # Convert 'datetime' column to datetime format
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    
    # Filter data for the year 2020
    dataset_2019 = dataset[dataset['datetime'].dt.year == 2019]

    # Calculate monthwise mean closing prices
    monthwise_mean = dataset_2019.groupby(dataset_2019['datetime'].dt.strftime('%B'))['close'].mean()
    monthwise_mean = monthwise_mean.reindex(new_order, axis=0)

    # Calculate monthwise original closing prices
    monthwise_original = dataset_2019.groupby(dataset_2019['datetime'].dt.strftime('%B'))['close'].last()
    monthwise_original = monthwise_original.reindex(new_order, axis=0)

    # Add traces to the figure with different colors for mean and original
    fig.add_trace(go.Bar(
        x=monthwise_mean.index,
        y=monthwise_mean,
        name=f'{file} Mean Closing Price',
        marker_color=color,
        opacity=0.7  # Adjust opacity for better visibility of both bars
    ))

    fig.add_trace(go.Bar(
        x=monthwise_original.index,
        y=monthwise_original,
        name=f'{file} Original Closing Price',
        marker_color=color,
        opacity=0.5  # Adjust opacity for better visibility of both bars
    ))

# Update layout and show the figure
fig.update_layout(barmode="group",
                  title='Monthwise Closing Prices of Bitcoin (Mean and Original) - 2019',
                  xaxis_title='Month',
                  yaxis_title='Closing Price')

fig.show()


# In[26]:


import plotly.express as px
from itertools import cycle

# Assuming 'datetime' is already in datetime format
maindf['open'] = maindf['close'].shift(1)
maindf['open'].fillna(maindf['close'][0], inplace=True)

# Filter data for a specific year (e.g., 2018)
year_filter = (maindf['datetime'] >= '2019-01-01') & (maindf['datetime'] < '2019-12-31')
filtered_data = maindf.loc[year_filter]

# Display 'volume' in the printed output for the filtered data
print(filtered_data[['datetime', 'open', 'high', 'low', 'close', 'volume']].head())

# Plotting using Plotly Express for the filtered data
names = cycle(['Bitcoin Open Price', 'Bitcoin Close Price', 'Bitcoin High Price', 'Bitcoin Low Price', 'Bitcoin Volume'])

fig = px.line(filtered_data, x=filtered_data['datetime'], y=[filtered_data['open'], filtered_data['close'], filtered_data['high'], filtered_data['low'], filtered_data['volume']],
              labels={'datetime': 'datetime', 'value': 'Stock Value'})

fig.update_layout(title_text='Bitcoin Analysis Chart (2019)', font_size=15, font_color='black',
                  legend_title_text='Bitcoin Parameters')

fig.for_each_trace(lambda t: t.update(name=next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

fig.show()


# # Analysis of 2020 

# In[27]:


maindf['datetime'] = pd.to_datetime(maindf['datetime'], format='%Y-%m-%d %H:%M:%S')  # Adjust the format as needed

y_2020 = maindf.loc[(maindf['datetime'] >= '2020-01-01') & (maindf['datetime'] < '2021-01-01')]

# Drop specified columns and assign the result back to y_2019
y_2020 = y_2020.drop(['volume'], axis=1)

"""

the code prepares the data for analysis by converting the date column to a datetime format,
Filtering the data to include only the year 2019
removing the 'Adj Close' and 'Volume' columns from the filtered data. 
The final result is stored in the DataFrame y_2020

"""


# In[28]:


import pandas as pd

# Define the list of dataset file names
dataset_files = ["btc_1h.csv", "btc_2h.csv", "btc_3m.csv", "btc_4h.csv", "btc_5m.csv", "btc_6h.csv", "btc_15m.csv", "btc_30m.csv"]

# Loop through each dataset file
for file in dataset_files:
    # Load the data
    dataset = pd.read_csv(file)  
    
    # Convert 'datetime' column to datetime format
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    
    # Filter data for the year 2020
    dataset_2020 = dataset[dataset['datetime'].dt.year == 2020]
    
    # Set 'datetime' column as the index
    dataset_2020.set_index('datetime', inplace=True)

    # Extract the time interval from the file name
    time_interval = file.split('_')[-1].split('.')[0]

    # Print the last five rows for each dataset in 2020
    print(f"Last five rows for {time_interval} in 2020:")
    print(dataset_2020.tail())
    print("\n" + "="*50 + "\n")  # Separate results for each dataset


# In[29]:


import pandas as pd

def preprocess_dataset(dataset, time_interval='1H', year=None):
    # Add a condition to filter data for a specific year
    if year is not None:
        dataset = dataset[dataset.index.year == year]

    # Assuming 'Close' column is the closing prices in the dataset
    dataset['open'] = dataset['close'].shift(1)

    # Fill NaN in the first row with the initial closing price (you may choose another strategy)
    dataset['open'].fillna(dataset['close'][0], inplace=True)

    # Filter data based on the provided time interval
    dataset_filtered = dataset.resample(time_interval).last()

    # Drop NaN rows that might occur due to resampling
    dataset_filtered = dataset_filtered.dropna()

    # Group by month and calculate mean
    monthwise = dataset_filtered.groupby(dataset_filtered.index.month)[['open', 'close']].mean()

    # Ensure all months are present in the results
    all_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    monthwise = monthwise.reindex(all_months)

    # Rename the index to month names
    monthwise.index = monthwise.index.map({1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                                           7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'})

    # Add year information to the results
    monthwise['Year'] = year

    return monthwise

# Define the list of dataset file names
dataset_files = ["btc_1h.csv", "btc_2h.csv", "btc_3m.csv", "btc_4h.csv", "btc_5m.csv", "btc_6h.csv", "btc_15m.csv", "btc_30m.csv"]

# Loop through each dataset file
for file in dataset_files:
    # Load the data
    dataset = pd.read_csv(file)  
    
    # Set 'datetime' column as the index
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    dataset.set_index('datetime', inplace=True)

    # Extract the time interval from the file name
    time_interval = file.split('_')[-1].split('.')[0]

    # Preprocess the data for the year 2018
    result = preprocess_dataset(dataset, time_interval=time_interval, year=2020)

    # Print the result
    print(f"Results for {time_interval} in 2020:")
    print(result)
    print("\n" + "="*50 + "\n")  # Separate results for each dataset


# In[30]:


import plotly.graph_objects as go
import pandas as pd

# Define the order of months
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# List of dataset file names
dataset_files = ["btc_1h.csv", "btc_2h.csv", "btc_3m.csv", "btc_4h.csv", "btc_5m.csv", "btc_6h.csv", "btc_15m.csv", "btc_30m.csv"]

# Define colors for each dataset
colors = ['rgb(0, 128, 0)', 'rgb(255, 165, 0)', 'rgb(0, 0, 128)', 'rgb(128, 0, 0)', 'rgb(128, 0, 128)', 'rgb(0, 128, 128)', 'rgb(255, 0, 0)', 'rgb(0, 255, 0)']

# Create an empty figure
fig = go.Figure()

# Loop through each dataset file
for file, color in zip(dataset_files, colors):
    # Load the data
    dataset = pd.read_csv(file)
    
    # Convert 'datetime' column to datetime format
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    
    # Filter data for the year 2020
    dataset_2020 = dataset[dataset['datetime'].dt.year == 2020]

    # Calculate monthwise mean closing prices
    monthwise_mean = dataset_2020.groupby(dataset_2020['datetime'].dt.strftime('%B'))['close'].mean()
    monthwise_mean = monthwise_mean.reindex(new_order, axis=0)

    # Add traces to the figure with different colors
    fig.add_trace(go.Bar(
        x=monthwise_mean.index,
        y=monthwise_mean,
        name=f'{file} Mean Closing Price',
        marker_color=color
    ))

# Update layout and show the figure
fig.update_layout(barmode="group",
                  title='Monthwise Mean Closing Price of Bitcoin (All Datasets) - 2020',
                  xaxis_title='Month',
                  yaxis_title='Mean Closing Price')

fig.show()

# mean is being calculated using the closing price for the bar graph


# In[31]:


# comparison between the mean and the original closing price 
import plotly.graph_objects as go
import pandas as pd

# Define the order of months
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# List of dataset file names
dataset_files = ["btc_1h.csv", "btc_2h.csv", "btc_3m.csv", "btc_4h.csv", "btc_5m.csv", "btc_6h.csv", "btc_15m.csv", "btc_30m.csv"]

# Define colors for each dataset
colors = ['rgb(0, 128, 0)', 'rgb(255, 165, 0)', 'rgb(0, 0, 128)', 'rgb(128, 0, 0)', 'rgb(128, 0, 128)', 'rgb(0, 128, 128)', 'rgb(255, 0, 0)', 'rgb(0, 255, 0)']

# Create an empty figure
fig = go.Figure()

# Loop through each dataset file
for file, color in zip(dataset_files, colors):
    # Load the data
    dataset = pd.read_csv(file)
    
    # Convert 'datetime' column to datetime format
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    
    # Filter data for the year 2020
    dataset_2020 = dataset[dataset['datetime'].dt.year == 2020]

    # Calculate monthwise mean closing prices
    monthwise_mean = dataset_2020.groupby(dataset_2020['datetime'].dt.strftime('%B'))['close'].mean()
    monthwise_mean = monthwise_mean.reindex(new_order, axis=0)

    # Calculate monthwise original closing prices
    monthwise_original = dataset_2020.groupby(dataset_2020['datetime'].dt.strftime('%B'))['close'].last()
    monthwise_original = monthwise_original.reindex(new_order, axis=0)

    # Add traces to the figure with different colors for mean and original
    fig.add_trace(go.Bar(
        x=monthwise_mean.index,
        y=monthwise_mean,
        name=f'{file} Mean Closing Price',
        marker_color=color,
        opacity=0.7  # Adjust opacity for better visibility of both bars
    ))

    fig.add_trace(go.Bar(
        x=monthwise_original.index,
        y=monthwise_original,
        name=f'{file} Original Closing Price',
        marker_color=color,
        opacity=0.5  # Adjust opacity for better visibility of both bars
    ))

# Update layout and show the figure
fig.update_layout(barmode="group",
                  title='Monthwise Closing Prices of Bitcoin (Mean and Original) - 2020',
                  xaxis_title='Month',
                  yaxis_title='Closing Price')

fig.show()


# In[32]:


import plotly.express as px
from itertools import cycle

# Assuming 'datetime' is already in datetime format
maindf['open'] = maindf['close'].shift(1)
maindf['open'].fillna(maindf['close'][0], inplace=True)

# Filter data for a specific year (e.g., 2020)
year_filter = (maindf['datetime'] >= '2020-01-01') & (maindf['datetime'] < '2020-12-31')
filtered_data = maindf.loc[year_filter]

# Display 'volume' in the printed output for the filtered data
print(filtered_data[['datetime', 'open', 'high', 'low', 'close', 'volume']].head())

# Plotting using Plotly Express for the filtered data
names = cycle(['Bitcoin Open Price', 'Bitcoin Close Price', 'Bitcoin High Price', 'Bitcoin Low Price', 'Bitcoin Volume'])

fig = px.line(filtered_data, x=filtered_data['datetime'], y=[filtered_data['open'], filtered_data['close'], filtered_data['high'], filtered_data['low'], filtered_data['volume']],
              labels={'datetime': 'datetime', 'value': 'Stock Value'})

fig.update_layout(title_text='Bitcoin Analysis Chart (2020)', font_size=15, font_color='black',
                  legend_title_text='Bitcoin Parameters')

fig.for_each_trace(lambda t: t.update(name=next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

fig.show()


# # Analysis of 2021 

# In[33]:


maindf['datetime'] = pd.to_datetime(maindf['datetime'], format='%Y-%m-%d %H:%M:%S')  # Adjust the format as needed

y_2021 = maindf.loc[(maindf['datetime'] >= '2021-01-01') & (maindf['datetime'] < '2022-01-31')]

# Drop specified columns and assign the result back to y_2019
y_2021 = y_2021.drop(['volume'], axis=1)

"""

the code prepares the data for analysis by converting the date column to a datetime format,
Filtering the data to include only the year 2019
removing the 'Adj Close' and 'Volume' columns from the filtered data. 
The final result is stored in the DataFrame y_2020

"""


# In[34]:


import pandas as pd

# Define the list of dataset file names
dataset_files = ["btc_1h.csv", "btc_2h.csv", "btc_3m.csv", "btc_4h.csv", "btc_5m.csv", "btc_6h.csv", "btc_15m.csv", "btc_30m.csv"]

# Loop through each dataset file
for file in dataset_files:
    # Load the data
    dataset = pd.read_csv(file)  
    
    # Convert 'datetime' column to datetime format
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    
    # Filter data for the year 2021
    dataset_2021 = dataset[dataset['datetime'].dt.year == 2021]
    
    # Set 'datetime' column as the index
    dataset_2021.set_index('datetime', inplace=True)

    # Extract the time interval from the file name
    time_interval = file.split('_')[-1].split('.')[0]

    # Print the last five rows for each dataset in 2021
    print(f"Last five rows for {time_interval} in 2021:")
    print(dataset_2021.tail())
    print("\n" + "="*50 + "\n")  # Separate results for each dataset


# In[35]:


import pandas as pd

def preprocess_dataset(dataset, time_interval='1H', year=None):
    # Add a condition to filter data for a specific year
    if year is not None:
        dataset = dataset[dataset.index.year == year]

    # Assuming 'Close' column is the closing prices in the dataset
    dataset['open'] = dataset['close'].shift(1)

    # Fill NaN in the first row with the initial closing price (you may choose another strategy)
    dataset['open'].fillna(dataset['close'][0], inplace=True)

    # Filter data based on the provided time interval
    dataset_filtered = dataset.resample(time_interval).last()

    # Drop NaN rows that might occur due to resampling
    dataset_filtered = dataset_filtered.dropna()

    # Group by month and calculate mean
    monthwise = dataset_filtered.groupby(dataset_filtered.index.month)[['open', 'close']].mean()

    # Ensure all months are present in the results
    all_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    monthwise = monthwise.reindex(all_months)

    # Rename the index to month names
    monthwise.index = monthwise.index.map({1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                                           7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'})

    # Add year information to the results
    monthwise['Year'] = year

    return monthwise

# Define the list of dataset file names
dataset_files = ["btc_1h.csv", "btc_2h.csv", "btc_3m.csv", "btc_4h.csv", "btc_5m.csv", "btc_6h.csv", "btc_15m.csv", "btc_30m.csv"]

# Loop through each dataset file
for file in dataset_files:
    # Load the data
    dataset = pd.read_csv(file)  
    
    # Set 'datetime' column as the index
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    dataset.set_index('datetime', inplace=True)

    # Extract the time interval from the file name
    time_interval = file.split('_')[-1].split('.')[0]

    # Preprocess the data for the year 2018
    result = preprocess_dataset(dataset, time_interval=time_interval, year=2021)

    # Print the result
    print(f"Results for {time_interval} in 2021:")
    print(result)
    print("\n" + "="*50 + "\n")  # Separate results for each dataset


# In[36]:


import plotly.graph_objects as go
import pandas as pd

# Define the order of months
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# List of dataset file names
dataset_files = ["btc_1h.csv", "btc_2h.csv", "btc_3m.csv", "btc_4h.csv", "btc_5m.csv", "btc_6h.csv", "btc_15m.csv", "btc_30m.csv"]

# Define colors for each dataset
colors = ['rgb(0, 128, 0)', 'rgb(255, 165, 0)', 'rgb(0, 0, 128)', 'rgb(128, 0, 0)', 'rgb(128, 0, 128)', 'rgb(0, 128, 128)', 'rgb(255, 0, 0)', 'rgb(0, 255, 0)']

# Create an empty figure
fig = go.Figure()

# Loop through each dataset file
for file, color in zip(dataset_files, colors):
    # Load the data
    dataset = pd.read_csv(file)
    
    # Convert 'datetime' column to datetime format
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    
    # Filter data for the year 2020
    dataset_2021 = dataset[dataset['datetime'].dt.year == 2021]

    # Calculate monthwise mean closing prices
    monthwise_mean = dataset_2020.groupby(dataset_2020['datetime'].dt.strftime('%B'))['close'].mean()
    monthwise_mean = monthwise_mean.reindex(new_order, axis=0)

    # Add traces to the figure with different colors
    fig.add_trace(go.Bar(
        x=monthwise_mean.index,
        y=monthwise_mean,
        name=f'{file} Mean Closing Price',
        marker_color=color
    ))

# Update layout and show the figure
fig.update_layout(barmode="group",
                  title='Monthwise Mean Closing Price of Bitcoin (All Datasets) - 2021',
                  xaxis_title='Month',
                  yaxis_title='Mean Closing Price')

fig.show()

# mean is being calculated using the closing price for the bar graph


# In[37]:


# comparison between the mean and the original closing price 
import plotly.graph_objects as go
import pandas as pd

# Define the order of months
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# List of dataset file names
dataset_files = ["btc_1h.csv", "btc_2h.csv", "btc_3m.csv", "btc_4h.csv", "btc_5m.csv", "btc_6h.csv", "btc_15m.csv", "btc_30m.csv"]

# Define colors for each dataset
colors = ['rgb(0, 128, 0)', 'rgb(255, 165, 0)', 'rgb(0, 0, 128)', 'rgb(128, 0, 0)', 'rgb(128, 0, 128)', 'rgb(0, 128, 128)', 'rgb(255, 0, 0)', 'rgb(0, 255, 0)']

# Create an empty figure
fig = go.Figure()

# Loop through each dataset file
for file, color in zip(dataset_files, colors):
    # Load the data
    dataset = pd.read_csv(file)
    
    # Convert 'datetime' column to datetime format
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    
    # Filter data for the year 2020
    dataset_2021 = dataset[dataset['datetime'].dt.year == 2021]

    # Calculate monthwise mean closing prices
    monthwise_mean = dataset_2021.groupby(dataset_2021['datetime'].dt.strftime('%B'))['close'].mean()
    monthwise_mean = monthwise_mean.reindex(new_order, axis=0)

    # Calculate monthwise original closing prices
    monthwise_original = dataset_2021.groupby(dataset_2021['datetime'].dt.strftime('%B'))['close'].last()
    monthwise_original = monthwise_original.reindex(new_order, axis=0)

    # Add traces to the figure with different colors for mean and original
    fig.add_trace(go.Bar(
        x=monthwise_mean.index,
        y=monthwise_mean,
        name=f'{file} Mean Closing Price',
        marker_color=color,
        opacity=0.7  # Adjust opacity for better visibility of both bars
    ))

    fig.add_trace(go.Bar(
        x=monthwise_original.index,
        y=monthwise_original,
        name=f'{file} Original Closing Price',
        marker_color=color,
        opacity=0.5  # Adjust opacity for better visibility of both bars
    ))

# Update layout and show the figure
fig.update_layout(barmode="group",
                  title='Monthwise Closing Prices of Bitcoin (Mean and Original) - 2021',
                  xaxis_title='Month',
                  yaxis_title='Closing Price')

fig.show()


# In[38]:


import plotly.express as px
from itertools import cycle

# Assuming 'datetime' is already in datetime format
maindf['open'] = maindf['close'].shift(1)
maindf['open'].fillna(maindf['close'][0], inplace=True)

# Filter data for a specific year (e.g., 2021)
year_filter = (maindf['datetime'] >= '2021-01-01') & (maindf['datetime'] < '2022-12-31')
filtered_data = maindf.loc[year_filter]

# Display 'volume' in the printed output for the filtered data
print(filtered_data[['datetime', 'open', 'high', 'low', 'close', 'volume']].head())

# Plotting using Plotly Express for the filtered data
names = cycle(['Bitcoin Open Price', 'Bitcoin Close Price', 'Bitcoin High Price', 'Bitcoin Low Price', 'Bitcoin Volume'])

fig = px.line(filtered_data, x=filtered_data['datetime'], y=[filtered_data['open'], filtered_data['close'], filtered_data['high'], filtered_data['low'], filtered_data['volume']],
              labels={'datetime': 'datetime', 'value': 'Stock Value'})

fig.update_layout(title_text='Bitcoin Analysis Chart (2021)', font_size=15, font_color='black',
                  legend_title_text='Bitcoin Parameters')

fig.for_each_trace(lambda t: t.update(name=next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

fig.show()


# # Overall Analysis from 2018 - 2022 

# In[39]:


maindf['datetime'] = pd.to_datetime(maindf['datetime'], format='%Y-%m-%d')

y_overall = maindf.loc[(maindf['datetime'] >= '2018-01-01') & (maindf['datetime'] <= '2022-01-31')]

# Drop the 'Volume' column
y_overall.drop(['volume'], axis=1, inplace=True)


# In[40]:


monthwise= y_overall.groupby(y_overall['datetime'].dt.strftime('%B'))[['open','close']].mean()
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
             'September', 'October', 'November', 'December']
monthwise = monthwise.reindex(new_order, axis=0)
monthwise


# In[41]:


maindf['datetime'] = pd.to_datetime(maindf['datetime'], format='%Y-%m-%d')

y_overall = maindf.copy()  # You may need to modify this based on your specific requirements

names = cycle(['Bitcoin Opening Price', 'Bitcoin Closing Price', 'Bitcoin Highest Price', 'Bitcoin Lowest Price'])

fig = px.line(y_overall, x=y_overall['datetime'], y=[y_overall['open'], y_overall['close'],
                                                 y_overall['high'], y_overall['low']],
              labels={'datetime': 'Date', 'value': 'Stock value'})
fig.update_layout(title_text='Bitcoin/USDT Analysis Chart', font_size=15, font_color='black',
                  legend_title_text='Stock Parameters')
fig.for_each_trace(lambda t: t.update(name=next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

fig.show()


# In[42]:


# Lets First Take all the Close Price 
closedf = maindf[['datetime','close']]
print("Shape of close dataframe:", closedf.shape)


# In[43]:


start_date = '2018-01-01'
end_date = '2022-01-31'

closedf = closedf[(closedf['datetime'] >= start_date) & (closedf['datetime'] <= end_date)]
close_stock = closedf.copy()

print("Total data for prediction:", closedf.shape[0])


# In[44]:


closedf


# In[45]:


fig = px.line(closedf, x=closedf.datetime, y=closedf.close,labels={'Date':'datetime','close':'close Stock'})
fig.update_traces(marker_line_width=2, opacity=0.8, marker_line_color='orange')
fig.update_layout(title_text='Considered period to predict Bitcoin close price', 
                  plot_bgcolor='white', font_size=15, font_color='black')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[46]:


# deleting date column and normalizing using MinMax Scaler

del closedf['datetime']
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
print(closedf.shape)


# In[47]:


# we keep the training set as 80% and 20% testing set

training_size=int(len(closedf)*0.80)
test_size=len(closedf)-training_size
train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]
print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)


# In[48]:


# convert an array of values into a dataset matrix

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[49]:


time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)


# In[50]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)


# # Actual Model Building 

# In[51]:


model=Sequential()
model.add(LSTM(10,input_shape=(None,1),activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error",optimizer="adam")


# In[ ]:


history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=200,batch_size=32,verbose=1)


# In[ ]:


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()

plt.show()


# In[ ]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
train_predict.shape, test_predict.shape


# # Model Evaluation

# In[ ]:


# Transform back to original form

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 


# In[ ]:


# Evaluation metrices RMSE and MAE
print("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain,train_predict)))
print("Train data MSE: ", mean_squared_error(original_ytrain,train_predict))
print("Train data MAE: ", mean_absolute_error(original_ytrain,train_predict))
print("-------------------------------------------------------------------------------------")
print("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest,test_predict)))
print("Test data MSE: ", mean_squared_error(original_ytest,test_predict))
print("Test data MAE: ", mean_absolute_error(original_ytest,test_predict))


# In[ ]:


# Variance Regression Score 

print("Train data explained variance regression score:", 
      explained_variance_score(original_ytrain, train_predict))
print("Test data explained variance regression score:", 
      explained_variance_score(original_ytest, test_predict))


# In[ ]:


# R score Regression 

print("Train data R2 score:", r2_score(original_ytrain, train_predict))
print("Test data R2 score:", r2_score(original_ytest, test_predict))


# - # Regression Loss Mean Gamma deviance regression loss (MGD) and Mean Poisson deviance regression loss (MPD)

# In[ ]:


print("Train data MGD: ", mean_gamma_deviance(original_ytrain, train_predict))
print("Test data MGD: ", mean_gamma_deviance(original_ytest, test_predict))
print("----------------------------------------------------------------------")
print("Train data MPD: ", mean_poisson_deviance(original_ytrain, train_predict))
print("Test data MPD: ", mean_poisson_deviance(original_ytest, test_predict))


# # Comparison of Bitcoin closing price and predicted close price 

# In[ ]:


# shift train predictions for plotting

look_back=time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
print("Train predicted data: ", trainPredictPlot.shape)

# shift test predictions for plotting
testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
print("Test predicted data: ", testPredictPlot.shape)

names = cycle(['Original close price','Train predicted close price','Test predicted close price'])


plotdf = pd.DataFrame({'datetime': close_stock['datetime'],
                       'original_close': close_stock['close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

fig = px.line(plotdf,x=plotdf['datetime'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                          plotdf['test_predicted_close']],
              labels={'value':'Stock price','datetime': 'datetime'})
fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[ ]:


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i=i+1
               
print("Output of predicted next days: ", len(lst_output))


# # Now lets plot the past 15 days and the next 30 days 

# In[ ]:


last_days=np.arange(1,time_step+1)
day_pred=np.arange(time_step+1,time_step+pred_days+1)
print(last_days)
print(day_pred)


# In[ ]:


temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

new_pred_plot = pd.DataFrame({
    'last_original_days_value':last_original_days_value,
    'next_predicted_days_value':next_predicted_days_value
})

names = cycle(['Last 15 days close price','Predicted next 30 days close price'])

fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                      new_pred_plot['next_predicted_days_value']],
              labels={'value': 'Stock price','index': 'Timestamp'})
fig.update_layout(title_text='Compare last 15 days vs next 30 days',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')

fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# When the graph is displayed, click on the last 15 days close price on the right side of the graph to display the last 15 days 
# Click on the red one for predicting the next 30 days close price 


# In[ ]:


lstmdf=closedf.tolist()
lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
lstmdf=scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]

names = cycle(['Close price'])

fig = px.line(lstmdf,labels={'value': 'Stock price','index': 'Timestamp'})
fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')

fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[ ]:


import plotly.express as px
import pandas as pd
from itertools import cycle

# Original closing prices
original_close = close_stock['close']

# Predicted closing prices
predicted_close = lstmdf[:len(original_close)]

# Creating a DataFrame for plotting
plotdf = pd.DataFrame({'datetime': close_stock['datetime'],
                       'original_close': original_close,
                       'predicted_close': predicted_close})

# Plotting
names = cycle(['Original Close Price', 'Predicted Close Price'])
fig = px.line(plotdf, x='datetime', y=['original_close', 'predicted_close'], labels={'value': 'Stock price', 'datetime': 'Timestamp'})
fig.update_layout(title_text='Comparison of Original and Predicted Closing Prices',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Stock')

fig.for_each_trace(lambda t: t.update(name=next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[ ]:


# 1. Gross Profit
gross_profit = (maindf['close'] - maindf['open']) * maindf['volume']
total_gross_profit = gross_profit.sum()

# 3. Total Closed Trades
total_closed_trades = len(maindf)

# 10. Largest Losing Trade (in USDT)
largest_losing_trade = trades.loc[maindf['profit'].idxmin()]

# 11. Largest Winning Trade (in USDT)
largest_winning_trade = maindf.loc[trades['profit'].idxmax()]

# 12. Sharpe Ratio
average_return = maindf['profit'].mean()
risk_free_rate = 0.02  # Assuming a 2% annual risk-free rate
std_dev_return = maindf['profit'].std()
sharpe_ratio = (average_return - risk_free_rate) / std_dev_return

# 13. Sortino Ratio
negative_returns = maindf[maindf['profit'] < 0]['profit']
sortino_ratio = (average_return - risk_free_rate) / negative_returns.std()

# 14. Average Holding Duration per Trade
maindf['holding_duration'] = maindf['exit_timestamp'] - maindf['entry_timestamp']
average_holding_duration = maindf['holding_duration'].mean()

# 15. Max Dip and Average Dip in Running Trade
running_trades = maindf[trades['exit_timestamp'].isnull()]  # Assuming running trades have a null exit timestamp
running_trades['max_dip'] = maindf['close'].max() - running_trades['close']
max_dip = running_trades['max_dip'].max()
average_dip = running_trades['max_dip'].mean()

# 7. Average Winning Trade (in USDT)
average_winning_trade = winning_trades['profit'].mean()

# 8. Average Losing Trade (in USDT)
losing_trades = maindf[maindf['profit'] < 0]
average_losing_trade = losing_trades['profit'].mean()

# 4. Win Rate (Profitability %)
winning_trades = maindf[maindf['profit'] > 0]
win_rate = (len(winning_trades) / total_closed_trades) * 100"

# 5. Max Drawdown
equity_curve = maindf['equity_curve']  
drawdown = (equity_curve.cummax() - equity_curve) / equity_curve.cummax()
max_drawdown = drawdown.max()

# 6. Gross Loss
gross_loss = (maindf['open'] - maindf['close']) * maindf['volume']
total_gross_loss = gross_loss.sum()

# 9. Buy and Hold Return of BTC
btc_buy_and_hold_return = (maindf['close'].iloc[-1] - maindf['close'].iloc[0]) / maindf['close'].iloc[0]

# 2. Net Profit
# Assuming you have a column 'transaction_cost' for transaction costs
net_profit = total_gross_profit - maindf['transaction_cost'].sum()


# In[ ]:





# In[ ]:




