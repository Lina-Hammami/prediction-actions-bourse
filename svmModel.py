# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 06:34:35 2021

@author: hamem
"""
#!/usr/bin/env python
# coding: utf-8

# In[50]:


from sklearn.svm import SVR
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
sns.set()

# In[51]:

# stocker et afficher les données 
pathto = '/Users/hamem/Downloads/prediction-action-bourse/'
df = pd.read_csv(pathto+'stock_prices.csv')
prix_actuel =  df.tail(1)
#prix_actuel

# preparer les données pour trainer les modèles SVR (on va avoir trois modèles ici )
# extraire toutes les données sauf la dernière ligne 
df = df.head(len(df)-1)
# une partie du dataset sera trainée et testée
df


# In[52]:


#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Historique des prix cloturés')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Prix de cloture ne USD ($)',fontsize=18)
plt.show()


# In[53]:


df = df[2500:2516]


# In[54]:


# création d'une liste vide pour les données dépendantes et indépendantes 
days = list() 
adj_close_prices =  list()

# obtenir les dates et les prix de clôture ajustés (adj_close_prices)
df_days = df.loc[:,'Date']
df_adj_close = df.loc[:,'Adj Close']


# In[55]:


# pour appler cette procedure dans le boucle de créations de days 
# days.append([int(date_to_timestamp(day))])
def date_to_timestamp(d): # non utilisée --> à améliorer plus tard 
    return  pd.to_datetime(d,format='%Y-%m-%d').timestamp()


# In[56]:


# création des données indépendantes 
for day in df_days: 
    # pour extraire les jours à partir des dates
    days.append([int(day.split('-')[2])]) 
    
# Création de dataset dépendante
for adj_close_price in df_adj_close:
    adj_close_prices.append(float(adj_close_price))

# In[57]:


# créer les 3 modèles SVR (SVM regresseur)

# créer et trainer un modèle svr en uilisant un kernel linéaire 
lin_svr = SVR(kernel='linear', C=100.0)
lin_svr.fit(days, adj_close_prices)

# créer et trainer un modèle svr en uilisant un kernel polinomial 
pol_svr = SVR(kernel='poly', C=100.0, degree=2)
pol_svr.fit(days, adj_close_prices)

# créer et trainer un modèle svr en uilisant un kernel rbf  
rbf_svr = SVR(kernel='rbf', C=100.0, gamma=0.15)
rbf_svr.fit(days, adj_close_prices)


# In[58]:


# plotter les modèles en un graph pour savoir qui a le meilleur entrainement pour la donnée originale 
plt.figure(figsize=(16,8))
plt.scatter(days, adj_close_prices, color='red', label='Data')
plt.plot(days, lin_svr.predict(days), color='blue', label='Linear Model')
plt.plot(days, pol_svr.predict(days), color='orange', label='Polynomial Model')
plt.plot(days, rbf_svr.predict(days), color='green', label='RBF Model')
plt.legend()
plt.show()


# In[59]:


# Afficher les prix prédits pour un jour donné 
day =[[19]]
print('Prédiction avec un SVR Linéaire :', lin_svr.predict(day) )
print('Prédiction avec un SVR Polynomial :', pol_svr.predict(day) )
print('Prédiction avec un SVR RBF :', rbf_svr.predict(day) )


# In[60]:
# Calcul d'erreur

from sklearn.model_selection import train_test_split
# Split data
X = days
y = df_adj_close
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
y_pred = rbf_svr.predict(X_test)

import math  
from sklearn.metrics import mean_squared_error
print(math.sqrt(mean_squared_error(y_test, y_pred)/100))

# In[ ]:
