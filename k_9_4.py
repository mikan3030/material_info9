#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
# from scipy import stats
import pandas as pd


# In[32]:


np.random.seed(seed=0)
y = np.random.randn(20)


# In[33]:


np.random.seed(seed=1)
x = np.random.randn(500,20)
y = y.reshape(20,1)


# In[34]:


df = pd.DataFrame({"y": list(y)})


# In[35]:


for i in range(500):
    df["x"+str(i)] = x[i]


# In[36]:


df["y"] = df["y"].astype(float)


# In[37]:


# df.head()


# In[38]:


# df.corr()


# In[39]:


corr = df.corr()
corr = dict(corr.loc["y"][1:])


# In[40]:


corr_swap = {v: k for k ,v in corr.items()}


# In[ ]:





# In[41]:


params = []
for i in sorted(corr_swap,reverse=True)[:10]:
    params.append(corr_swap[i])


# In[42]:


# params


# In[43]:


from sklearn.linear_model import LinearRegression


# In[44]:


new_df = df[params]


# In[45]:


# new_df


# In[46]:


X = new_df.values
y = df["y"].values


# In[47]:


model = LinearRegression()
model.fit(X,y)


# In[48]:


print("スコア:"+ str(model.score(X,y))+"\n")


# In[51]:


# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
# kfold = KFold(n_splits =5)
# model_cross_val = LinearRegression()
# scores = cross_val_score(model_cross_val,X,y,cv=5)


# In[52]:


# scores


# In[53]:


kfold = KFold(n_splits = 5)
for train_ind ,test_ind in kfold.split(X):
    X_train,X_test = X[train_ind],X[test_ind]
    y_train,y_test = y[train_ind],y[test_ind]
#     print(y_train)
#    print(X_train)
    model_2 = LinearRegression()
    model_2.fit(X_train,y_train)
    score_train = model.score(X_train,y_train)
    score_test = model.score(X_test,y_test)
    print("trainのスコア:{}".format(score_train))
    print("testのスコア:{}\n".format(score_test))


# In[54]:


corr = df[:16].corr()
corr = dict(corr.loc["y"][1:])
corr_swap = {v: k for k ,v in corr.items()}
params = []
for i in sorted(corr_swap,reverse=True)[:10]:
    params.append(corr_swap[i])
# params


# In[55]:


new_df = df[params]
train_X = new_df.loc[:16][:]
train_y = df.loc[:16]["y"]
test_X = new_df.loc[16:][:]
test_y = df.loc[16:]["y"]


# In[56]:


model = LinearRegression()
model.fit(train_X,train_y)
print("testデータのスコア:{}".format(model.score(test_X,test_y)))

# In[60]:





# In[ ]:




