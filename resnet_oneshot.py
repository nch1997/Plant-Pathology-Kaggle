# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 14:51:21 2020

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

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
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
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.callbacks import LearningRateScheduler

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
IMAGE_SIZE = 512
#need to reduce batch size cause GPU not strong enough 
train_batch_size = 4 #normal practice is 32 batch
val_batch_size = 4


image_gen_train = ImageDataGenerator(preprocessing_function = preprocess_input,
                                     rotation_range=10,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     fill_mode='nearest',
                                     brightness_range=[0.5, 1.5])

image_gen_valid = ImageDataGenerator(preprocessing_function = preprocess_input)

#flow_from_dataframe/directory will directly convert the picture to specific image size

train_gen = image_gen_train.flow_from_dataframe(df_train , 
                                                directory ='D:/Projects/Plan Pathology (Categorical)/plant-pathology-2020-fgvc7/images',
                                                x_col = 'image_id', #column with input
                                                y_col = ['healthy', 'multiple_diseases', 'rust', 'scab'], #column with answers
                                                class_mode = 'raw',
                                                target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                                batch_size = train_batch_size,
                                                shuffle = True)

val_gen = image_gen_valid.flow_from_dataframe(df_val , 
                                                directory ='D:/Projects/Plan Pathology (Categorical)/plant-pathology-2020-fgvc7/images',
                                                x_col = 'image_id', #column with input
                                                y_col = ['healthy', 'multiple_diseases', 'rust', 'scab'], #column with answers
                                                class_mode = 'raw',
                                                target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                                batch_size = train_batch_size,
                                                shuffle = True)

test_gen = image_gen_valid.flow_from_dataframe(df_test , 
                                                directory ='D:/Projects/Plan Pathology (Categorical)/plant-pathology-2020-fgvc7/images',
                                                x_col = 'image_id', #column with input
                                                y_col = ['healthy', 'multiple_diseases', 'rust', 'scab'], #column with answers
                                                class_mode = 'raw',
                                                target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                                batch_size = 1,
                                                shuffle = False)

#learning rate scheduler function
def build_lrfn(lr_start=0.00001, lr_max=0.000075, lr_min=0.000001, lr_rampup_epochs=20, lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max 
    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    
    return lrfn

lrfn = build_lrfn()


#%%
#creating model
resnet_path = r'C:\Users\User\.keras\models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
dropout_dense = 0.5

#if nvr sbase_model.trainable = False everything is true so fine tune from the start
base_model = ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet')

#check if unfreezed
for i, layer in enumerate(base_model.layers):
   print(i, layer.name, layer.trainable)
   

#creating model
model = Sequential()
#exclude prediction layer from pretrain model include_top = False
model.add(base_model)

#flatten 
model.add(Flatten())


#full connection
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(dropout_dense))
model.add(Dense(18, activation='relu'))
model.add(Dropout(dropout_dense))
model.add(Dense(4, activation = "softmax"))

# Compile the model
model.compile(optimizer = 'adam', 
              loss = "categorical_crossentropy", 
              metrics=["accuracy"])


model.summary()


epochs = 50


#stop the training if no improvements
earlystopper = EarlyStopping(monitor='val_loss', 
                             patience=3, 
                             verbose=1, #1 to make statement when it triggers 0 for silent (no statement)
                             restore_best_weights=True)
#save model every epoch
checkpoint = ModelCheckpoint('RESNET50_oneshot.h5', 
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             save_weights_only=False, 
                             mode='auto', period=1)

#reduce learning rate if no improvement
reduce_learn_rate = ReduceLROnPlateau(monitor='val_loss', 
                                      factor = 0.3, 
                                      patience = 3, 
                                      mode = 'auto', 
                                      verbose = 1,
                                      min_lr=0.000001)

#change learning rate based on a function created by user
lr_schedule = LearningRateScheduler(lrfn, verbose = 1)

history = model.fit_generator(train_gen,
                              steps_per_epoch=int(np.ceil(1082 / float(train_batch_size))),
                              epochs=epochs,
                              validation_data=val_gen,
                              validation_steps=int(np.ceil(465 / float(val_batch_size))),
                              callbacks = [checkpoint, lr_schedule])




#%%
#plotting results
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


model.save('D:/Projects/Plan Pathology (Categorical)/plant-pathology-2020-fgvc7/RESNET50_oneshot_95.h5')

