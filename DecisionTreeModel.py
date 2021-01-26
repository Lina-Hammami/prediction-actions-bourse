# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 01:48:19 2021

@author: hamem
"""
#Import pandas to import a CSV file:
import pandas as pd

pathto = '/Users/hamem/Downloads/prediction-action-bourse/'
dataset = pd.read_csv(pathto+'stock_prices.csv')
dataset = dataset[-200:]

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

# Creating the decision tree regressor model
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor().fit(xtrain, ytrain)


xfuture = dataset.drop(["Prediction"], 1)[:-futureDays]
xfuture = xfuture.tail(futureDays)
xfuture = np.array(xfuture)
print(xfuture)


#To see the model tree prediction
treePrediction = tree.predict(xfuture)
print("Prediction avec Arbre de décision =",treePrediction)

import matplotlib.pyplot as plt
#Visualize decision tree predictions
predictions = treePrediction
valid = dataset[x.shape[0]:]
valid["Predictions"] = predictions
plt.figure(figsize=(10, 6))
plt.title("Prediction des prix de course de l'indice S&P500 (Modèle régresseur de l'arbre de décision)")
plt.xlabel("Jours")
plt.ylabel("Prix de cloture en USD ($)")
plt.plot(dataset["Close"])
plt.plot(valid[["Close", "Predictions"]])
plt.legend(["Original", "Valid", "Predictions"])
plt.show()

import math  
from sklearn.metrics import mean_squared_error
ypred = tree.predict(xtest)
math.sqrt(mean_squared_error(ytest, ypred)/1000)