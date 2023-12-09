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
Final_all_dataset = pd.read_csv (r"new_surge_no_inter_no_miss.csv")
Final_all_dataset= Final_all_dataset.drop (columns= ["Unnamed: 0"])
Final_all_dataset


# In[ ]:


X = Final_all_dataset.iloc[:, :12]
y= Final_all_dataset.iloc[:, 12:]


# In[ ]:


#def custom_loss(y_true, y_pred):
    #n = tf.shape(y_true) 
    #loss = (tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred)))) / (2 * tf.cast(n, tf.float32))
    #return loss

remaining_indices = list(range(0, (70578631)))
fold11_indices = remaining_indices[(0):(27087851)]
fold2_indices = remaining_indices[(27087851):(30940375)]
fold3_indices = remaining_indices[(30940375):(34989276)]
fold4_indices = remaining_indices[(34989276):(39233323)]
fold5_indices = remaining_indices[(39233323):(43672579)]
fold6_indices = remaining_indices[(43672579):(48302890)]
fold7_indices = remaining_indices[(48302890):(52238348)]
fold8_indices = remaining_indices[(52238348):(56428966)]
fold9_indices = remaining_indices[(56428966):(60877013)]
fold10_indices =remaining_indices[(60877013):(65600377)]
fold1_indices = remaining_indices[(65600377):(70578631)]

fold_indices = [fold1_indices, fold2_indices, fold3_indices,
                fold4_indices, fold5_indices,fold6_indices,
                fold7_indices, fold8_indices, fold9_indices, fold10_indices,fold11_indices]



kf = KFold(n_splits=11, shuffle=False)
for i, (train_indices, test_indices) in enumerate(kf.split(fold_indices)):

    current_train_indices = [idx for fold_idx in train_indices for idx in fold_indices[fold_idx]]
    current_test_indices = [idx for fold_idx in test_indices for idx in fold_indices[fold_idx]]
    
    X_train, y_train = np.array(X)[current_train_indices], np.array(y)[current_train_indices]
    X_test, y_test = np.array(X)[current_test_indices], np.array(y)[current_test_indices]
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = keras.Sequential()

    model.add(layers.Dense(256,input_shape=(12,), activation='relu'))

    model.add(layers.Dense(256, activation='relu'))

    model.add(layers.Dense(256, activation='relu'))

    model.add(layers.Dense(256, activation='relu'))

    model.add(Dense(1, activation='linear'))
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="mean_squared_error",
              optimizer=optimizer, metrics=['mean_absolute_error'])
    model.summary()
    print ("the shape of X_train is:", X_train.shape)
    print ("the shape of X_test is:", X_test.shape)

    model.fit(X_train,y_train,verbose=1, batch_size= 2500,
              epochs=100)
    print("start")
    start_time = time.time()  # Record the start time

    pred = model.predict(X_test)
    end_time = time.time()  # Record the end time
    print("finish")
    pd.DataFrame (pred).to_csv (f'{i}_{"Record_Prediction"}.csv')
    pd.DataFrame (y_test).to_csv (f'{i}_{"Record_Observation"}.csv')

    # Measure each fold's RMSE
    score = np.sqrt(metrics.mean_squared_error(pred,y_test))
    print(f"Fold score (RMSE): {score}")
    del pred
    del y_test


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

