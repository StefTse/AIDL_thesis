#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

from sklearn.metrics import classification_report

# Load data
Emo_Data_polarity = pd.read_csv('Emo_Data_polarity.csv')

print(Emo_Data_polarity.columns)
print('')
print(Emo_Data_polarity.shape)
print(Emo_Data_polarity.head())
print('')
Emo_Data_polarity.shape


# In[2]:


# Replacing intent text values to numerical
Emo_Data_polarity=Emo_Data_polarity.replace({"neutral":0,"negative":1,"positive":2, "ambiguous":3})
Emo_Data_polarity['Emotion'].unique()


# In[3]:


# Split dataset into train, validation, and test sets
x_train, x_test, y_train, y_test = train_test_split(Emo_Data_polarity["Text"], Emo_Data_polarity["Emotion"],
                                                    test_size=0.1, shuffle=True, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True, random_state=1)
x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape


# In[ ]:


from transformers import XLNetTokenizer, TFXLNetForSequenceClassification

# Tokenize and pad
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

x_train_pad = tokenizer.batch_encode_plus(x_train.tolist(), return_tensors='tf', padding=True, truncation=True, max_length=170 )
x_val_pad = tokenizer.batch_encode_plus(x_val.tolist(), return_tensors='tf',padding=True, truncation=True,   max_length=170)
x_test_pad = tokenizer.batch_encode_plus(x_test.tolist(), return_tensors='tf', padding=True, truncation=True, max_length=17 0)


# In[23]:


# Convert BatchEncoding to NumPy arrays
x_train_array = {key: x_train_pad[key] for key in x_train_pad}
x_val_array = {key: x_val_pad[key] for key in x_val_pad}


# In[24]:


# One-hot-encode classes
n_classes = len(np.unique(y_train))
y_train_np = y_train.values
y_val_np = y_val.values
y_test_np = y_test.values

y_train_enc = tf.keras.utils.to_categorical(y_train_np, n_classes)
y_val_enc = tf.keras.utils.to_categorical(y_val_np, n_classes)
y_test_enc = tf.keras.utils.to_categorical(y_test_np, n_classes)


# In[25]:


#datasets check
print(x_val_pad['input_ids'].shape)
print(y_val_enc.shape)
empty_sequences = np.sum(x_val_pad['input_ids'] == 0, axis=1) == 512
print(np.any(empty_sequences))
print(np.any(np.isnan(x_val_pad['input_ids'])))
print(np.any(np.isnan(y_val_enc)))

print(x_train_pad['input_ids'].shape)
print(y_train_enc.shape)
empty_sequences = np.sum(x_train_pad['input_ids'] == 0, axis=1) == 512
print(np.any(empty_sequences))
print(np.any(np.isnan(x_train_pad['input_ids'])))
print(np.any(np.isnan(y_train_enc)))


# In[26]:


#Create model
from transformers import TFDistilBertForSequenceClassification
model_xlnet = TFXLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=4)

for layer in model_xlnet.layers[:-2]:
     layer.trainable = False


# In[27]:


#from tensorflow.keras.callbacks import EarlyStopping

#compile model
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5)
loss = tf.keras.losses.CategoricalCrossentropy()
metric = tf.keras.metrics.CategoricalAccuracy()
#keras_callbacks = [EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0.0001)]
model_xlnet.compile(optimizer=optimizer, loss=loss, metrics=metric)


# In[ ]:


# train model with callbacks.

# history = model_xlnet.fit(x_train_array, y_train_enc, batch_size=16, epochs=10,
#                          validation_data=(x_val_array, y_val_enc),
#                          callbacks=[keras_callbacks])

history = model_xlnet.fit(x_train_array, y_train_enc, epochs=3, batch_size=16, 
                          validation_data=(x_val_array, y_val_enc))


# In[ ]:




