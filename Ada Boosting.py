#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


df = pd.read_csv("pima-indians-diabetes.csv")


# In[4]:


df.head()


# In[5]:


df.isnull().sum()


# In[6]:


import seaborn as sns


# In[7]:


sns.distplot(df["Glucose"])


# In[8]:


df.describe()


# In[9]:


sns.distplot(df["Age"])


# In[10]:


df["Pregnancies"].value_counts()


# In[11]:


df.shape


# In[12]:


df["Pregnancies"].unique()


# In[13]:


df["Pregnancies"].nunique()


# In[14]:


df.dtypes


# In[15]:


df["Pregnancies"] = df["Pregnancies"].astype("category")
df["Outcome"] = df["Outcome"].astype("category")


# In[16]:


df["Age"] = df["Age"].apply(np.log1p)


# In[17]:


df["Insulin"] = df["Insulin"].apply(np.log1p)


# In[18]:


df_num = df.select_dtypes(include =["float64", "int64"])
df_cat = df.select_dtypes(exclude =["float64", "int64"])


# In[19]:


df_cat = df_cat.drop("Outcome",axis=1)


# In[20]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[21]:


mn = MinMaxScaler()
df_num_sc = mn.fit_transform(df_num)


# In[22]:


df_num_df = pd.DataFrame(df_num_sc, index = df_num.index, columns=df_num.columns)


# In[23]:


df_cat_dum = pd.get_dummies(df_cat)


# In[24]:


df_final = pd.concat([df_num_df, df_cat_dum],axis=1)


# In[25]:


df_final.head()


# In[26]:


x = df_final
y = df["Outcome"]


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state = 10)


# In[29]:


from sklearn.ensemble import AdaBoostClassifier


# In[30]:


ad = AdaBoostClassifier()
ad.fit(x_train, y_train)


# In[31]:


pred = ad.predict(x_test)


# In[32]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[33]:


accuracy_score(y_test, pred)


# In[34]:


accuracy_score(y_train, ad.predict(x_train))


# In[35]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state = 100)


# In[36]:


ad = AdaBoostClassifier()
ad.fit(x_train, y_train)


# In[37]:


pred = ad.predict(x_test)


# In[38]:


accuracy_score(y_test, pred)


# In[39]:


accuracy_score(y_train, ad.predict(x_train))


# In[40]:


ad = AdaBoostClassifier(n_estimators=500)
ad.fit(x_train, y_train)


# In[41]:


pred = ad.predict(x_test)


# In[42]:


accuracy_score(y_test, pred)


# In[43]:


accuracy_score(y_train, ad.predict(x_train))


# In[44]:


from sklearn.model_selection import cross_validate


# In[45]:


ad_cv = cross_validate(ad, x,y, cv = 10, return_train_score=True)


# In[46]:


ad_test = np.average(ad_cv["test_score"])


# In[47]:


ad_train = np.average(ad_cv["train_score"])


# In[48]:


ad_test


# In[49]:


ad_train


# In[ ]:




