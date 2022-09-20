#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st


# In[2]:


games=pd.read_csv("steam_games.csv")


# In[3]:


games.head (5)


# In[4]:


games.original_price


# In[5]:


games['original_price'] = games['original_price'].str[1:]


# In[6]:


games['original_price'] = pd.to_numeric(games['original_price'],errors='coerce')


# In[7]:


games[(games['original_price'] >= 10.00) & (games['original_price'] <= 60.00)]


# In[8]:


games.shape


# In[9]:


games.head(3)


# In[10]:


games = games[['genre','game_details','popular_tags','developer','name']]


# In[11]:


games.head(3)


# In[12]:


games.dropna(inplace = True)


# In[13]:


games.shape


# In[14]:


games['game_ID'] = range(0,37114)


# In[15]:


games.isnull().values.any()


# In[16]:


games = games.reset_index()


# In[17]:


def get_important_features(data):
    important_features = []
    for i in range(0, data.shape[0]):
        important_features.append(data['name'][i]+' '+data['developer'][i]+' '+data['popular_tags'][i]+' '+data['genre'][i]+data['game_details'][i])
        
    return important_features


# In[18]:


games['important_features'] = get_important_features(games)
games.important_features.head(5)


# In[19]:


cm = CountVectorizer().fit_transform(games['important_features'])


# In[20]:


cs = cosine_similarity(cm)


# In[21]:


print(cs)


# In[22]:


title = 'Call of Duty®: Black Ops II'
title_id = games[games.name == title]['game_ID'].values[0]


# In[23]:


scores = list(enumerate(cs[title_id]))


# In[24]:


sorted_scores = sorted(scores, key = lambda x:x[1], reverse = True)
sorted_scores = sorted_scores[1:]


# In[25]:


a = 0
print('The 7 most recommended games to', title, 'are:\n')
for item in sorted_scores:
    game_title = games[games.game_ID == item[0]]['name'].values[0]
    print(a+1, game_title)
    a = a+1
    if a > 6:
        break


# In[26]:


games = games.set_index('name')


# In[27]:


games.loc[['Call of Duty®: Black Ops II - Apocalypse','Call of Duty®: Black Ops II - Uprising','Call of Duty®: Ghosts','Call of Duty®: Black Ops III','Call of Duty®: Black Ops III - Season Pass','Left 4 Dead 2','Call of Duty®: Black Ops'],
         ['genre','game_details','popular_tags','developer']]


# In[28]:


import pickle
pickle.dump(games, open("games.pkl","wb"))


# In[29]:


loaded_model = pickle.load(open('games.pkl','rb'))


# In[ ]:




