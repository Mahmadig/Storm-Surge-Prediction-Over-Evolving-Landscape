#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
from sklearn import metrics
from scipy.stats import zscore
from keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras import layers


# In[ ]:


####
import pandas as pd
import numpy as np
X = pd.read_csv (r"storm_param.csv")
X= X.drop (columns= ["Unnamed: 0"])
X


# In[ ]:


y = pd.read_csv (r"surge_values.csv")
y= y.drop (columns= ["Unnamed: 0"])
y


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


#def custom_loss(y_true, y_pred):
    #n = tf.shape(y_true) 
    #loss = (tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred)))) / (2 * tf.cast(n, tf.float32))
    #return loss


scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = keras.Sequential()

model.add(layers.Dense(256,input_shape=(5,), activation='relu'))

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(256, activation='relu'))

model.add(Dense(126174, activation='linear'))
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="mean_squared_error",
              optimizer=optimizer, metrics=['mean_absolute_error'])
model.summary()


history = model.fit(X_train_scaled, y_train, batch_size= 100 ,validation_split=0.2, epochs =1000, shuffle= True)
print("start")
start_time = time.time()  # Record the start time

pred = model.predict(X_test_scaled)
end_time = time.time()  # Record the end time
print("finish")

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['mean_absolute_error']
val_acc = history.history['val_mean_absolute_error']
plt.plot(epochs, acc, 'y', label='Training MAE')
plt.plot(epochs, val_acc, 'r', label='Validation MAE')
plt.title('Training and validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


# calculate the Pearson's correlation between
from numpy.random import randn
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from numpy.random import seed
from scipy.stats import pearsonr

corr, _ = pearsonr(np.array(pred.iloc[:]).reshape(-1), np.array(y_tes.iloc[:]).reshape(-1))
print('Pearsons correlation: %.3f' % corr)
mae_ANN = mean_absolute_error(np.array(pred.iloc[:]).reshape(-1), np.array(y_tes.iloc[:]).reshape(-1))
print('Mean absolute error Using ANN: ', mae_ANN)

