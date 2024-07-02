#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, \
confusion_matrix, classification_report


from sklearn.datasets import load_wine


# In[6]:


url = 'wine.csv'
df = pd.read_csv(url)
df.head()


# In[7]:


from sklearn.datasets import load_wine

Alcohol = load_wine()

Alcohol.keys()

X = Alcohol['data']
y = Alcohol['target']

print(X.shape)
print(y.shape)


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[9]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)


# In[10]:


y_pred = model.predict(X_test)
y_pred


# In[11]:


import pandas as  pd

df = pd.DataFrame(X,columns=Alcohol['feature_names'])

df['target'] = y
df.sample()


# In[12]:


Alcohol['target_names']


# In[13]:


from sklearn.metrics import accuracy_score,\
confusion_matrix,\
classification_report

cm = confusion_matrix(y_test, y_pred)
cm


# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm,annot=True)
plt.show()


# In[15]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[16]:


cr = classification_report(y_test, y_pred)
print(cr)


# In[ ]:




