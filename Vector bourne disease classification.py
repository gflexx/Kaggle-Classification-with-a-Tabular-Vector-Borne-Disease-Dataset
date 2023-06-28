#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


csv = pd.read_csv('train.csv')
csv.head()


# In[3]:


csv.columns


# In[4]:


csv.shape


# In[5]:


csv.dtypes


# In[6]:


# check for NaN
csv.isna().sum()


# In[8]:


# drop the target column
test = csv.drop(["prognosis", "id"], axis=1)
test.columns


# In[9]:


# select and create the target collumn
target = csv["prognosis"]


# In[10]:


target


# In[11]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from decimal import Decimal


# In[12]:


# split for testing and training using training data to get accuracy
X_test, X_train, y_test, y_train = train_test_split(test, target, test_size=0.27)


# In[13]:


def get_accuracy(y_test, prediction):
    accuracy_rf = round(
    Decimal(
            accuracy_score(y_test, prediction) * 100
        ), 2
    )
    return accuracy_rf


# In[14]:


# create RandomForestClassifier
rf = RandomForestClassifier(n_estimators=603, criterion='entropy')
rf = rf.fit(test, target)
rf_pred = rf.predict(X_test)
accuracy_rf = get_accuracy(y_test, rf_pred)
print(f"RF accuracy: {accuracy_rf}")


# In[15]:


# create KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9, weights='distance', n_jobs=4)
knn = knn.fit(test, target)
knn_pred = knn.predict(X_test)
accuracy_knn = get_accuracy(y_test, knn_pred)
print(f"KNN accuracy: {accuracy_rf}")


# In[16]:


dat_test = pd.read_csv('test.csv')
dat_test.head()


# In[17]:


dat_test.columns


# In[18]:


dat_test.shape


# In[19]:


# prepare test data for predictions
x__test = dat_test.drop(['id'], axis=1)

columns = x__test.columns
predictions = []
for i in x__test.values:
    df = pd.DataFrame.from_records([i])
    df.columns = columns
    rf_pred = rf.predict(df)
    predictions.append(rf_pred)


# In[26]:


id_list = []
prediction_list = []
for i, j in enumerate(predictions):
    id_list.append(
        round(dat_test.iloc[i].id)
    )
    prediction_list.append(j[0])


# In[27]:


df_columns = ["id", "prognosis"]
df_new = pd.DataFrame(columns=df_columns)
df_new["id"] = id_list
df_new["prognosis"] = prediction_list


# In[28]:


df_new.head()


# In[29]:


df_new.to_csv('submission.csv', index=False , header = 1)


# In[30]:


df_new.shape


# In[ ]:




