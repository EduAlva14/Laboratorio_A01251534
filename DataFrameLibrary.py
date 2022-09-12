#!/usr/bin/env python
# coding: utf-8

# In[148]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math


# In[149]:


df = pd.read_csv("House_Rent_Dataset.csv", sep=",",index_col=False)
df = df[["Size","Rent"]]
train, test = train_test_split(df, test_size=0.2)
train.head()


# In[150]:



plt.scatter(train["Size"],train["Rent"]);


# In[151]:


x = train["Size"]
x.head()


# In[152]:


y = train["Rent"]
y.head()


# In[153]:


w = test["Size"]
w.head()


# In[154]:


z = test["Rent"]
z.head()


# In[155]:


model = LinearRegression(fit_intercept=True)
model


# In[156]:


X = x[:, np.newaxis]
X.shape


# In[157]:


model.fit(X, y)


# In[158]:


A = model.coef_ # This is the parameter value


# In[159]:


B = model.intercept_ # This is the bias value


# In[160]:


xfit = np.linspace(-1, 11)


# In[161]:


Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)


# In[162]:


plt.scatter(x, y)
plt.plot(xfit, yfit, color = "red");


# In[163]:


Rent = A*w + B


# In[164]:


Rent


# In[173]:


mse = np.sum((Rent-z)*(Rent-z)) / 950


# In[174]:


mse


# In[175]:


mape = np.sum((Rent - z)/z) / 950


# In[176]:


mape


# 

# 

# In[169]:





# In[ ]:





# 
