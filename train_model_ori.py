# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:10:59 2020

@author: User
"""

import os
import shutil

import tensorflow as tf
import numpy as np
import pandas as pd
print(tf.__version__)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

#%%
#formatting and visualization
train = pd.read_csv('D:/Projects/Plan Pathology (Categorical)/plant-pathology-2020-fgvc7/train.csv')

#add .jpg behind image_id

train['image_id'] = train['image_id'] + '.jpg'
train.head()

healthy = train.healthy.value_counts()
multiple_diseases = train.multiple_diseases.value_counts()
rust = train.rust.value_counts()
scab = train.scab.value_counts()

labels = ['healthy', 'multiple diseases', 'rust', 'scab']
label_count = [healthy[1], multiple_diseases[1], rust[1], scab[1]]
explode = [0.1, 0, 0, 0]
fig1, ax1 = plt.subplots()
ax1.pie(label_count, explode = explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Ratio of Leaf Diseases')
plt.show()


fig2, ax2 = plt.subplots()
ax2.barh(labels, label_count)
plt.title('Ratio of Leaf Diseases')
plt.xlabel('Counts')
plt.show()

#%%
#preprocessing data

y = ['healthy', 'multiple_diseases', 'rust', 'scab']
df_train_val, df_test = train_test_split(train, test_size = 0.15, 
                                         random_state = 101)
df_train, df_val = train_test_split(df_train_val, test_size = 0.30, 
                                    random_state = 101)
IMAGE_SIZE = 240 
train_batch_size = 32 #normal practice is 32 batch
val_batch_size = 32


image_gen_train = ImageDataGenerator(rescale=1./255,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')

image_gen_valid = ImageDataGenerator(rescale = 1./255)

#flow_from_dataframe/directory will directly convert the picture to specific image size

train_gen = image_gen_train.flow_from_dataframe(df_train , 
                                                directory ='D:/Projects/Plan Pathology (Categorical)/plant-pathology-2020-fgvc7/images',
                                                x_col = 'image_id', #column with input
                                                y_col = ['healthy', 'multiple_diseases', 'rust', 'scab'], #column with answers
                                                class_mode = 'raw',
                                                target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                                batch_size = train_batch_size,
                                                shuffle = False)

val_gen = image_gen_train.flow_from_dataframe(df_val , 
                                                directory ='D:/Projects/Plan Pathology (Categorical)/plant-pathology-2020-fgvc7/images',
                                                x_col = 'image_id', #column with input
                                                y_col = ['healthy', 'multiple_diseases', 'rust', 'scab'], #column with answers
                                                class_mode = 'raw',
                                                target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                                batch_size = train_batch_size,
                                                shuffle = False)

test_gen = image_gen_train.flow_from_dataframe(df_test , 
                                                directory ='D:/Projects/Plan Pathology (Categorical)/plant-pathology-2020-fgvc7/images',
                                                x_col = 'image_id', #column with input
                                                y_col = ['healthy', 'multiple_diseases', 'rust', 'scab'], #column with answers
                                                class_mode = 'raw',
                                                target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                                batch_size = 1,
                                                shuffle = False)

#%%
#train model

#parameters to CNN
kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128
fourth_filters = 256
fifth_filters = 512

#dropout stops nodes from learning at random rate to ensure they don rely on 
#only 1 node
dropout_conv = 0.3
dropout_dense = 0.5

#Initializing the CNN
model = Sequential()

#1st layer
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

#2nd layer
model.add(Conv2D(second_filters, kernel_size, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

#3rd layer
model.add(Conv2D(third_filters, kernel_size, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

#4th layer
model.add(Conv2D(fourth_filters, kernel_size, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

#5th layer
model.add(Conv2D(fifth_filters, kernel_size, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

#flatten 
model.add(Flatten())

#full connection
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(dropout_dense))
model.add(Dense(32, activation='relu'))
#althought binary is chosen the sigmoid turn the asnwer tovalues between 0 to 1
#the model then use threshold to determine 0 or 1
#e.g <0.5 = 0 >0.5 = 1
model.add(Dense(4, activation = "softmax"))

#for alot classes
#softmax gives probability of each belong to which class
#for example with 3 classes
#given image
#ans : [0.6,0.2,0.2] mean first class is answer
#model.add(Dense(num_classes, activation = "softmax"))

# Compile the model
model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics=["accuracy"])
#for more classes
#model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()
#training the model
epochs = 200

#stop the training if no improvements
earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
#save model every epoch
checkpoint = ModelCheckpoint('checkpoint_model_ori.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

history = model.fit_generator(train_gen,
                              steps_per_epoch=int(np.ceil(1082 / float(train_batch_size))),
                              epochs=epochs,
                              validation_data=val_gen,
                              validation_steps=int(np.ceil(465 / float(val_batch_size))),
                              callbacks = [checkpoint])
#callbacks = [earlystopper, checkpoint])


#%%
# only use if run until all epoch finish (cannot use with early stopping)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(val_loss))

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#%%

prediction = model.evaluate_generator(generator = test_gen, steps = 274)