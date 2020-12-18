import sys
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import InputLayer, Reshape, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils

image_size = 256
num_labels = 8

df = pd.read_csv('data/dataset.csv')

X_train,Y_train,X_test,Y_test=[],[],[],[]

for index, row in df.iterrows():
    
    val=row['pixels'].split(" ")
    
    try:
        if 'Training' in row['usage']:
           X_train.append(np.array(val,'float32'))
           Y_train.append(row['emotion'])
        elif 'Testing' in row['usage']:
           X_test.append(np.array(val,'float32'))
           Y_test.append(row['emotion'])
    except:
        print('error occured at index {}'.format(index))


X_train = np.array(X_train,'float32')
Y_train = np.array(Y_train,'float32')
X_test = np.array(X_test,'float32')
Y_test = np.array(Y_test,'float32')

Y_train=np_utils.to_categorical(Y_train, num_classes=num_labels)
Y_test=np_utils.to_categorical(Y_test, num_classes=num_labels)

#normalizing data between oand 1
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

X_train = X_train.reshape(X_train.shape[0], image_size, image_size, 1)
X_test = X_test.reshape(X_test.shape[0], image_size, image_size, 1)

##designing the cnn
#1st convolution layer
model = Sequential()

model.add(InputLayer(input_shape=(X_train.shape[1:])))
model.add(Reshape((image_size,image_size,1)))
model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same', activation='relu'))
model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))

#2nd convolution layer
model.add(Conv2D(kernel_size=10, strides=1, filters=64, padding='same', activation='relu'))
model.add(Conv2D(kernel_size=10, strides=1, filters=64, padding='same', activation='relu'))
model.add(Conv2D(kernel_size=10, strides=1, filters=64, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(MaxPooling2D(pool_size=2, strides=2))

#3rd convolution layer
model.add(Conv2D(kernel_size=15, strides=1, filters=128, padding='same', activation='relu'))
model.add(Conv2D(kernel_size=15, strides=1, filters=128, padding='same', activation='relu'))
model.add(Conv2D(kernel_size=15, strides=1, filters=128, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))

#4th convolution layer
model.add(Conv2D(kernel_size=20, strides=1, filters=256, padding='same', activation='relu'))
model.add(Conv2D(kernel_size=20, strides=1, filters=256, padding='same', activation='relu'))
model.add(Conv2D(kernel_size=20, strides=1, filters=256, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Flatten())

#fully connected neural networks
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(8, activation='softmax'))

# model.summary()

#Compliling the model

model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-5), 
							loss='categorical_crossentropy',
              metrics=['accuracy'])

#Training the model
model.fit(X_train, Y_train,
          batch_size=96,
          epochs=30,
          verbose=1,
          validation_data=(X_test, Y_test),
          shuffle=True)

emotion_json = model.to_json()
with open("data/model.json", "w") as json_file:
    json_file.write(emotion_json)
model.save_weights("data/weight.h5")