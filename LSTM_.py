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


# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# In[13]:


X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)


# In[14]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[15]:


# train the model
def build_model(X_train, y_train):
	# prepare data
	# X_train, y_train = train[:,0], train[:,1] # REMOVED
	# define parameters
	verbose, epochs, batch_size = 1, 150, 16
	n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], 1 # Output always 1 here and you can't get it from the shape

	# define model
	model = Sequential()
	model.add(LSTM(128, activation='relu', input_shape=(n_timesteps, n_features), return_sequences=True))
	model.add(LSTM(64, activation='relu', return_sequences=True))
	model.add(LSTM(32, activation='relu'))
	model.add(Dense(16, activation='relu'))
	model.add(Dense(n_outputs))
	model.compile(loss='mse', optimizer='adam')
	model.summary()
	# fit network
	model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model


# In[16]:


model=build_model(X_train, y_train)


# In[17]:


prediction = model.predict(X_test)
prediction


# In[18]:


from sklearn.metrics import mean_squared_error
import math


# In[19]:


# Get the root mean squared error
math.sqrt(mean_squared_error(y_test, prediction))


# In[20]:


days = ['mon', 'tue', 'wed', 'thr', 'fri', 'sat' , 'sun']
plt.xlabel('days')
pyplot.plot(y_test,)
pyplot.plot(prediction)
pyplot.show()

