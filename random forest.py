#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import numpy as np
import io
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM


# In[2]:


# load the new file
datasett = read_csv('./FB_.csv', header=0)


# In[3]:


datasett.head()


# In[4]:


dataset=datasett[['date','adj_close']]


# In[5]:


dataset.head()


# In[6]:


dataset.shape


# In[7]:


import matplotlib.pyplot as plt
from matplotlib import dates as mpl_dates
from datetime import datetime, timedelta
plt.plot_date(dataset['date'],dataset['adj_close'],linestyle='solid')
plt.xlabel('date')
plt.ylabel('close_value')
plt.title('Stock market')
plt.show()


# In[8]:


# split into train and test
from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset.values)


# In[9]:


# we will create a dataset where X are the real values and y are the next day values (the forecasting to be learned by the model)
def create_dataset(dataset, look_back=1):
	X, y = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 1]
		X.append(a)
		y.append(dataset[i + look_back, 1])
	return array(X), array(y)


# In[10]:


# Create a training & testing data where X are the real values and y are the next day values
X_train, y_train = create_dataset(train)
X_test, y_test = create_dataset(test)


# In[11]:


X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)


# In[12]:


X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)


# In[14]:


from sklearn.ensemble import RandomForestRegressor
tree=RandomForestRegressor(n_estimators=100).fit(X_train, y_train)


# In[15]:


prediction = tree.predict(X_test)
prediction


# In[16]:


from sklearn.metrics import mean_squared_error
import math


# In[17]:


# Get the root mean squared error
math.sqrt(mean_squared_error(y_test, prediction))


# In[18]:


days = ['mon', 'tue', 'wed', 'thr', 'fri', 'sat' , 'sun']
plt.xlabel('days')
pyplot.plot(y_test,)
pyplot.plot(prediction)
pyplot.show()

