#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


df = pd.read_csv('processed_training_dataset.csv')


# In[5]:


df


# In[8]:


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


# In[13]:


X = df.drop('TARGET' , axis = 1)
y = df['TARGET']


# In[11]:


neg = len(df[df['TARGET'] == 0])
pos = len(df[df['TARGET'] == 1])
scale = neg / pos


# In[15]:


n_estimators = [300 , 500 , 600]
learning_rate = [0.05 , 0.1]
max_depth = [4 , 5 , 6 , 8]
subsample = [0.85]
colsample_bytree = [0.8]

param_grid = {'n_estimators' : n_estimators , 'learning_rate' : learning_rate , 'max_depth' : max_depth , 'subsample' : subsample , 'colsample_bytree' : colsample_bytree , 'scale_pos_weight' : [scale]}

model = XGBClassifier(objective = 'binary:logistic' , eval_metric = 'auc' , n_jobs = -1)

grid_model = GridSearchCV(estimator = model , param_grid = param_grid , n_jobs = -1 , scoring = 'roc_auc' , cv = 6)
grid_model.fit(X , y)


# In[16]:


grid_model.best_params_


# In[250]:


xgb_model = XGBClassifier(colsample_bytree = 0.8 , learning_rate = 0.0475 , max_depth = 4, n_estimators = 612, scale_pos_weight = scale, subsample = 0.85 , min_child_weight = 1.13 , gamma = 1.6 , n_jobs=-1)
xgb_model.fit(X , y)


# In[ ]:




