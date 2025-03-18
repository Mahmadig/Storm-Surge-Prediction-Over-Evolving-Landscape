####

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ReduceLROnPlateau
from keras.layers import BatchNormalization, Activation, Dropout
from sklearn import metrics
from keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import shap
## shap can be used to explain the output


def custom_loss(y_true, y_pred):
    n = tf.shape(y_true)
    loss = (tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred)))) / (2 * tf.cast(n, tf.float32))
    return loss

class CNN_MLP_Model:
    def __init__(self, rate, filter_size, stride_num):

        self.conv1 =      layers.Conv1D(filter_size, (stride_num))
        self.dropout1 =   layers.Dropout(rate)
        self.batchN1 =    layers.BatchNormalization()
        self.activation1= layers.Activation('relu')
        #self.pool1 =      layers.MaxPooling1D((1))
        #self.activation1= layers.PReLU()

        self.conv2 = layers.Conv1D(filter_size, (stride_num))
        self.dropout2 = layers.Dropout(rate)
        self.batchN2 =    layers.BatchNormalization()
        self.activation2= layers.Activation('relu')
        #self.pool2 =      layers.MaxPooling1D((1))
        #self.activation2= layers.PReLU()


        self.conv3 = layers.Conv1D(filter_size, (stride_num))
        self.dropout3 = layers.Dropout(rate)
        self.batchN3 =    layers.BatchNormalization()
        self.activation3= layers.Activation('relu')
        #self.pool2 =      layers.MaxPooling1D((1))
        #self.activation3= layers.PReLU()


        self.conv4 = layers.Conv1D(filter_size, (stride_num))
        self.dropout4 = layers.Dropout(rate)
        self.batchN4 =    layers.BatchNormalization()
        self.activation4= layers.Activation('relu')
        #self.pool2 =      layers.MaxPooling1D((1))
        #self.activation4= layers.PReLU()

        self.flatten = layers.Flatten()


        self.dense_mlp1 = layers.Dense(filter_size)
        self.batchN5 =    layers.BatchNormalization()
        self.dropout5 = layers.Dropout(rate)
        self.activation5= layers.Activation('relu')
        #self.activation5= layers.PReLU()


        self.dense_mlp2 = layers.Dense(filter_size)
        self.batchN6 =    layers.BatchNormalization()
        self.dropout6 = layers.Dropout(rate)
        self.activation6= layers.Activation('relu')
        #self.activation6= layers.PReLU()


        self.dense_mlp3 = layers.Dense(filter_size)
        self.batchN7 =    layers.BatchNormalization()
        self.dropout7 = layers.Dropout(rate)
        self.activation7= layers.Activation('relu')
        #self.activation7= layers.PReLU()


        self.dense_mlp4 = layers.Dense(filter_size)
        self.batchN8 =    layers.BatchNormalization()
        self.dropout8 = layers.Dropout(rate)
        self.activation8= layers.Activation('relu')
        #self.activation8= layers.PReLU()


        self.final_dense = layers.Dense(1, activation='linear')

    def build_model(self):
        cnn_input = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]), name="cnn_input")

        x = self.conv1(cnn_input)
        x = self.dropout1(x)
        x=  self.batchN1(x)
        x= self.activation1(x)
        #x = self.pool1(x)

        x = self.conv2(x)
        x = self.dropout2(x)
        x=  self.batchN2(x)
        x= self.activation2(x)
        #x = self.pool1(x)

        x = self.conv3(x)
        x = self.dropout3(x)
        x=  self.batchN3(x)
        x= self.activation3(x)
        #x = self.pool1(x)

        x = self.conv4(x)
        x = self.dropout4(x)
        x=  self.batchN4(x)
        x= self.activation4(x)
        #x = self.pool1(x)


        x = self.flatten(x)

        x = self.dense_mlp1(x)
        x=  self.batchN5(x)
        x = self.dropout5(x)
        x= self.activation5(x)

        x = self.dense_mlp2(x)
        x=  self.batchN6(x)
        x = self.dropout6(x)
        x= self.activation6(x)

        x = self.dense_mlp3(x)
        x=  self.batchN7(x)
        x = self.dropout7(x)
        x= self.activation7(x)

        x = self.dense_mlp4(x)
        x=  self.batchN8(x)
        x = self.dropout8(x)
        x= self.activation8(x)

        output = self.final_dense(x)
        # for the second mode Concatenate CNN and MLP outputs
        #y = self.dense_mlp1(mlp_input)
        # change the output of MLP to y and CNN to x
        #combined = layers.Concatenate(axis=1)([x, y])
        model = Model(inputs=cnn_input, outputs=output)

        return model


kf = KFold(n_splits=11, shuffle=False)

remaining_indices = list(range(0, (43490780+24559408)))
fold11_indices = remaining_indices[(0):(24559408)]
fold1_indices = remaining_indices[(24559408):(3852524+24559408)]
fold2_indices = remaining_indices[(3852524+24559408):(7901425+24559408)]
fold3_indices = remaining_indices[(7901425+24559408):(12145472+24559408)]
fold4_indices = remaining_indices[(12145472+24559408):(16584728+24559408)]
fold5_indices = remaining_indices[(16584728+24559408):(21215039+24559408)]
fold6_indices = remaining_indices[(21215039+24559408):(25150497+24559408)]
fold7_indices = remaining_indices[(25150497+24559408):(29341115+24559408)]
fold8_indices = remaining_indices[(29341115+24559408):(33789162+24559408)]
fold9_indices =remaining_indices[(33789162+24559408):(38512526+24559408)]
fold10_indices = remaining_indices[(38512526+24559408):(43490780+24559408)]

fold_indices = [fold1_indices, fold2_indices, fold3_indices,
                fold4_indices, fold5_indices,fold6_indices,
                fold7_indices, fold8_indices, fold9_indices, fold10_indices,fold11_indices]

### two options added to implement the skip conection and link to models in parallel. by default they are cascade

## for implementing skip conenction, follow these steps, but adjust the code accordingly


def Residual_model(input_layer, filter_number, kernel_size):

    pad_layer_zero_1 = ZeroPadding1D(padding=(kernel_size - 1))(input_layer)
    conv1D_layer_1 = Conv1D(filters=filter_number//4, kernel_size=1, padding='same',activation='relu')(pad_layer_zero_1)
    batchnorm_layer_1 = BatchNormalization()(conv1D_layer_1)
    drop_layer_1 = Dropout(drop)(batchnorm_layer_1)

    pad_layer_zero_2 = ZeroPadding1D(padding=(kernel_size - 1))(drop_layer_1)
    conv1D_layer_2 = Conv1D(filters=filter_number//4, kernel_size=kernel_size, padding='same',activation='relu')(pad_layer_zero_2)
    batchnorm_layer_2 = BatchNormalization()(conv1D_layer_2)
    drop_layer_2 = Dropout(drop)(batchnorm_layer_2)


    pad_layer_zero_3 =ZeroPadding1D(padding=(kernel_size - 1))(drop_layer_2)
    conv1D_layer_3 = Conv1D(filters=filter_number, kernel_size=1,  padding='same',activation='relu')(pad_layer_zero_3)
    batchnorm_layer_3 = BatchNormalization()(conv1D_layer_3)
    drop_layer_3 = Dropout(drop)(batchnorm_layer_3)

    pad_layer_zero_4 = ZeroPadding1D(padding=(kernel_size - 1))(drop_layer_3)
    conv1D_layer_4 = Conv1D(filters=filter_number, kernel_size=kernel_size,  padding='valid',activation='relu')(pad_layer_zero_4)
    batchnorm_layer_4 = BatchNormalization()(conv1D_layer_4)
    drop_layer_4 = Dropout(drop)(batchnorm_layer_4)


    RES = Conv1D(filter_number=filter_number, kernel_size=kernel_size, padding='same')(input_layer)
    RES_layer = Add()([RES, drop_layer_4])
    RES_layer = ReLU()(RES_layer)
    return RES_layer

def build_model(input_shape, filter_number):
    input_layer = Input(shape=input_shape)

    RES_layer = Residual_model(input_layer, filter_number=filter_number, kernel_size=kernel_size)
    #no attention needed since it increases the computional cost
    #attention = my_block_attention(Residual_model)

    #cnn_layer = Conv1D(filters=filter_number, kernel_size=kernel_size, padding='same', activation='relu')(attention)
    #cnn_layer = BatchNormalization()(cnn_layer)
    #cnn_layer = Conv1D(filters=filter_number , kernel_size=kernel_size, padding='same', activation='relu')(cnn_layer)
    #cnn_layer = BatchNormalization()(cnn_layer)
    Drop_out = Dropout(drop)(RES_layer)
    Drop_out= keras.layers.GlobalAveragePooling1D()(Drop_out)
    Drop_out = Dense(256, activation='relu')(Drop_out)
    Drop_out = Dense(256, activation='relu')(Drop_out)
    Drop_out = Dense(256, activation='relu')(Drop_out)
    Drop_out = Dense(256, activation='relu')(Drop_out)

    Drop_out_final = Dense(1, activation='linear')(Drop_out)

    model = Model(inputs=input_layer, outputs=Drop_out_final)
    return model

for i, (train_indices, test_indices) in enumerate(kf.split(fold_indices)):
    current_train_indices = [idx for fold_idx in train_indices for idx in fold_indices[fold_idx]]
    current_test_indices = [idx for fold_idx in test_indices for idx in fold_indices[fold_idx]]

    X_train, y_train = np.array(X)[current_train_indices], np.array(y)[current_train_indices]
    X_test, y_test = np.array(X)[current_test_indices], np.array(y)[current_test_indices]

    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape(X_train.shape[0],1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0],1, X_test.shape[1])

    cnn_mlp_model = CNN_MLP_Model( rate, filter_size, stride_num)
    model = cnn_mlp_model.build_model()
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience = 3, verbose=1 ,
                                            factor=0.75, min_lr=0.00001)
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=["RootMeanSquaredError"])

    model.summary()
    plot_model(model, to_file='cnn_mlp_model.png', show_shapes=True)

    model.fit(X_train,y_train,verbose=1 ,batch_size= batch_number, epochs=epochs_number, validation_split=(0.2), callbacks=[learning_rate_reduction] )
    pred = model.predict(X_test)
    # Measure this fold's RMSE
    score = np.sqrt(metrics.mean_squared_error(pred,y_test))
    print(f"Fold score (RMSE): {score}")

    explainer = shap.KernelExplainer(model.predict,X_train)
    shap_values = explainer.shap_values(X_test,nsamples=100)
    shap.summary_plot(shap_values,X_test,feature_names=X.columns.values)
