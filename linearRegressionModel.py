# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 06:33:12 2021

@author: hamem
"""
#Import pandas to import a CSV file:
import pandas as pd

pathto = '/Users/hamem/Downloads/prediction-action-bourse/'
dataset = pd.read_csv(pathto+'newest_stock_prices.csv')
print(dataset.head())
print("trainging days =",dataset.shape)
dataset = dataset[-200:]

#To Visualize the close price Data:
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.figure(figsize=(10, 4))
plt.title("Prix de course de bourse ")
plt.xlabel("Jours")
plt.ylabel("Prix de Cloture USD ($)")
plt.plot(dataset["Close"])
plt.show()

# To get the close price:
dataset = dataset[["Close"]]
print(dataset.head())
#Creating a variable to predict ‘X’ days in the future:
futureDays = 25
#Create a new target column shifted ‘X’ units/days up:
dataset["Prediction"] = dataset[["Close"]].shift(-futureDays)
print(dataset.head())
print(dataset.tail())
#To create a feature dataset (x) and convert into a numpy array and remove last ‘x’ rows/days:
import numpy as np
x = np.array(dataset.drop(["Prediction"], 1))[:-futureDays]
print(x)
#To create a target dataset (y) and convert it to a numpy array and get all of the target values except the last ‘x’ rows days:
y = np.array(dataset["Prediction"])[:-futureDays]
print(y)

#Split the data into 75% training and 25% testing
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)

# Creating the Linear Regression model
from sklearn.linear_model import LinearRegression
linear = LinearRegression().fit(xtrain, ytrain)
#To get the last ‘x’ rows/days of the feature dataset:

xfuture = dataset.drop(["Prediction"], 1)[:-futureDays]
xfuture = xfuture.tail(futureDays)
xfuture = np.array(xfuture)
print(xfuture)

#To see the model linear regression prediction
linearPrediction = linear.predict(xfuture)
print("Linear regression Prediction =",linearPrediction)

#Visualize the linear model predictions
predictions = linearPrediction
valid = dataset[x.shape[0]:]
valid["Predictions"] = predictions
plt.figure(figsize=(10, 6))
plt.title("Prediction des prix de course de l'indice S&P500 (Modèle de Regression Linéaire)")
plt.xlabel("Jours")
plt.ylabel("Prix de Cloture en USD ($)")
plt.plot(dataset["Close"])
plt.plot(valid[["Close", "Predictions"]])
plt.legend(["Original", "Valid", "Predictions"])
plt.show()

# Calcul RMSE
import math  
from sklearn.metrics import mean_squared_error
ypred = linear.predict(xtest)
math.sqrt(mean_squared_error(ytest, ypred)/1000)