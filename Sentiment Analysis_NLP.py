#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install textblob')


# In[2]:


import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
nltk.download('stopwords')
stop_words = stopwords.words('english')


# In[37]:


DATASET_COLUMNS = ['target', 'ids', 'date', 'status', 'user', 'text']
df = pd.read_csv(r'C:\Users\mohammad\training.1600000.processed.noemoticon.csv',encoding='ISO-8859-1', names=DATASET_COLUMNS)
df.head()


# In[38]:


df.info


# In[39]:


df = df.drop(['ids', 'date', 'status', 'user'], axis=1)
df.head


# In[40]:


import seaborn as sns
sns.countplot(data=df, x='target')


# In[16]:


stop_words = stopwords.words('english')
stop_words.remove('not')


# In[41]:


import re
import string
english_punctuations = string.punctuation
punctuations_list = english_punctuations


def cleaning_text(x):
    temp =" ".join([w for w in str(x).split() if w not in stop_words])
    translator =  str.maketrans("", "", punctuations_list)
    temp =  str(temp).translate(translator)
    temp = re.sub('((www.[^s]+)|(https?://[^s]+))',' ',temp)
    temp = re.sub('[0-9]+', '', temp)
    #  remove special characters
    temp = re.sub(r"[^a-zA-Z0-9]+", ' ', temp)
    return temp.lower()


# In[31]:


from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(max_features=1200)
vectorized_data = count_vectorizer.fit_transform(df['text']).toarray()

y= df['target']


# In[42]:


df.head


# In[43]:


from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(max_features=1200)
vectorized_data = count_vectorizer.fit_transform(df['text']).toarray()

y= df['target']


# In[44]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report


# In[45]:


X=df['text']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state =42)


# In[46]:


vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)


# In[47]:


X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)


# In[48]:


def model_Evaluate(model):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
    xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)


# In[49]:


BNBmodel = BernoulliNB()
BNBmodel.fit(X_train, y_train)
model_Evaluate(BNBmodel)
y_pred1 = BNBmodel.predict(X_test)

