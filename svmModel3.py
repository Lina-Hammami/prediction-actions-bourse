# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 00:15:04 2021

@author: hamem
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

pathto = '/Users/hamem/Downloads/prediction-action-bourse/'
dataset = pd.read_csv(pathto+'newest_stock_prices.csv')
#dataset = dataset[-200:]


# To get the close price:
dataset = dataset[["Close"]]
print(dataset.head())
#Creating a variable to predict ‘X’ days in the future:
futureDays = 25
# Création d'une nouvelle colonne décalée de 'futureDays' jours 
dataset["Prediction"] = dataset[["Close"]].shift(-futureDays)

#Generate sample data
X = np.array(dataset.drop(["Prediction"], 1))[:-futureDays]
y = np.array(dataset["Prediction"])[:-futureDays]


xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25)

#Add noise to targets
#y[::5] += 3 * (0.5 - np.random.rand(8))

#Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

#look at the results
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.hold('on')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()








"""
Mode d'utilisation basique de l'SVR (SVM pour la regression) 
extrait du scikitlearn documentation officielle 


print(__doc__)

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
#Generate sample data

X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

#Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

#Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

#look at the results
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.hold('on')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

"""