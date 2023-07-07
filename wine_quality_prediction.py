
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[19]:


wine=pd.read_csv('WineQT.csv')


# In[20]:


wine


# In[21]:


wine.describe()


# In[22]:


wine.info()


# In[6]:


wine.head()


# In[7]:


sns.heatmap(wine.corr(),annot=True)


# In[8]:


sns.displot(wine['quality'])


# In[9]:


wine.columns


# In[10]:


X=wine[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
Y=wine['quality']


# In[11]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=101) 


# In[12]:


from sklearn.linear_model import LinearRegression 

lm = LinearRegression() 

lm.fit(X_train,y_train) 


# In[13]:


predictions = lm.predict(X_test)


# In[14]:


plt.scatter(y_test,predictions)


# In[15]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions)) 
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions))) 


# In[17]:


lm.predict([[6.2,0.600,0.08,2.0,0.090,32.0,44.0,0.99490,3.45,0.58,10.5]])


# In[ ]:
