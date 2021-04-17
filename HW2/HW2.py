#!/usr/bin/env python
# coding: utf-8

# ## HW2: Linear Discriminant Analysis
# In hw2, you need to implement Fisher’s linear discriminant by using only numpy, then train your implemented model by the provided dataset and test the performance with testing data
# 
# Please note that only **NUMPY** can be used to implement your model, you will get no points by calling sklearn.discriminant_analysis.LinearDiscriminantAnalysis 

# ## Load data

# In[113]:


import numpy as np


# In[114]:


x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")


# In[115]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# ## 1. Compute the mean vectors mi, (i=1,2) of each 2 classes

# In[ ]:


## Your code HERE


# In[ ]:


print(f"mean vector of class 1: {m1}", f"mean vector of class 2: {m2}")


# ## 2. Compute the Within-class scatter matrix SW

# In[ ]:


## Your code HERE


# In[ ]:


assert sw.shape == (2,2)
print(f"Within-class scatter matrix SW: {sw}")


# ## 3.  Compute the Between-class scatter matrix SB

# In[ ]:


## Your code HERE


# In[ ]:


assert sb.shape == (2,2)
print(f"Between-class scatter matrix SB: {sb}")


# ## 4. Compute the Fisher’s linear discriminant

# In[ ]:


## Your code HERE


# In[ ]:


assert w.shape == (2,1)
print(f" Fisher’s linear discriminant: {w}")


# In[ ]:





# ## 5. Project the test data by linear discriminant to get the class prediction by nearest-neighbor rule and calculate the accuracy score 
# you can use accuracy_score function from sklearn.metric.accuracy_score

# In[ ]:


acc = accuracy_score(y_test, y_pred)


# In[ ]:


print(f"Accuracy of test-set {acc}")


# ## 6. Plot the 1) best projection line on the training data and show the slope and intercept on the title (you can choose any value of intercept for better visualization) 2) colorize the data with each class 3) project all data points on your projection line. Your result should look like [this image](https://i.imgur.com/tubMQpw.jpg)

# In[ ]:




