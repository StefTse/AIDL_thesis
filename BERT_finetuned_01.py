#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import modules and data
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report


Emotions_Data=pd.read_csv('Emotions_Data.csv')

print(Emotions_Data.columns)
print('')
print(Emotions_Data.shape)
print(Emotions_Data.head())
print('')
Emotions_Data.shape


# In[2]:


# Replacing intent text values to numerical 
Emotions_Data_num=Emotions_Data.replace({"sadness":0,"anger":1,"love":2, "surprise":3, "fear":4, 
                         "happiness":5, "neutral":6, "worry":7, "fun":8, "hate":9, "relief":10,"gratitude":11,"disapproval":12,
                          "amusement":13,"disappointment":14,"admiration":15, "realization":16,"annoyance":17,"confusion":18,"optimism":19,   
                          "curiosity":20, "caring":21,"approval":22, "joy":23, "excitement":24})
Emotions_Data_num['Emotion'].unique()


# In[3]:


# train, validation and test split dataset
x_train, x_test, y_train, y_test = train_test_split(Emotions_Data_num["Text"], Emotions_Data_num["Emotion"], test_size = 0.1, shuffle=True, random_state = 1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, shuffle=True, random_state = 1) 
x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape


# In[6]:


from transformers import BertTokenizer


# In[8]:


#tokenize and pad
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

x_train_pad = tokenizer.batch_encode_plus(x_train, return_tensors='tf', padding='max_length')
x_val_pad = tokenizer.batch_encode_plus(x_val, return_tensors='tf', padding='max_length')
x_test_pad = tokenizer.batch_encode_plus(x_test, return_tensors='tf', padding='max_length')


# In[5]:


# #tokenize and pad
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# max_length = 140
# x_train_pad = tokenizer.batch_encode_plus(x_train, return_tensors='np', padding='max_length')
# x_val_pad = tokenizer.batch_encode_plus(x_val, return_tensors='np', padding='max_length')
# x_test_pad = tokenizer.batch_encode_plus(x_test, return_tensors='np', padding='max_length')


# In[9]:


#one-hot-encoding of classes
n_classes = len(np.unique(y_train))
y_train_np = y_train.values
y_val_np = y_val.values
y_test_np=y_test.values

y_train_enc = tf.keras.utils.to_categorical(y_train_np, n_classes)
y_val_enc = tf.keras.utils.to_categorical(y_val_np, n_classes)
y_test_enc = tf.keras.utils.to_categorical(y_test_np, n_classes)


# In[10]:


# print(x_val_pad['input_ids'].shape)
# print(y_val_enc.shape)
# empty_sequences = np.sum(x_val_pad['input_ids'] == 0, axis=1) == 512
# print(np.any(empty_sequences))
# print(np.any(np.isnan(x_val_pad['input_ids'])))
# print(np.any(np.isnan(y_val_enc)))


# In[ ]:


# print(x_train_pad['input_ids'].shape)
# print(y_train_enc.shape)
# empty_sequences = np.sum(x_train_pad['input_ids'] == 0, axis=1) == 512
# print(np.any(empty_sequences))
# print(np.any(np.isnan(x_train_pad['input_ids'])))
# print(np.any(np.isnan(y_train_enc)))


# In[11]:


#create BERT model

from transformers import TFBertForSequenceClassification
model_BERT = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=25)
model_BERT.summary()



# In[12]:


#Freeze all layers except the classification layer
for layer in model_BERT.layers[:-1]:
    layer.trainable = False


# In[13]:


from tensorflow.keras.callbacks import EarlyStopping

#compile model
#optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5)#legacy Adam optimizer runs better on M1 macOS
loss = tf.keras.losses.CategoricalCrossentropy()
metric = tf.keras.metrics.CategoricalAccuracy()
keras_callbacks = [EarlyStopping(monitor='val_loss', patience=20, mode='min', min_delta=0.0001)]
model_BERT.compile(optimizer=optimizer, loss=loss, metrics=metric) #run_eagerly=True)


# In[14]:


# train model with callbacks. 

history = model_BERT.fit(x_train_pad, y_train_enc, batch_size=16, epochs=100, validation_data=(x_val_pad, y_val_enc), callbacks=[keras_callbacks])



# In[ ]:




