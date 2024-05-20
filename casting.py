#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:17:47 2024

@author: ai-1
"""

import os
import cv2
import time
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import AUC,Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
#from tensorflow.keras.applications import Xception,VGG19,ResNet50,InceptionResNetV2,ResNet152V2,ConvNeXtTiny 
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,GlobalMaxPooling2D,Dropout,Conv2D,Flatten,MaxPooling2D

random.seed(555)



#train

dir_train_ok ='casting/casting_data/casting_data/train/ok_front'
dir_train_def='casting/casting_data/casting_data/train/def_front'

#test

dir_test_ok='casting/casting_data/casting_data/test/ok_front'
dir_test_def='casting/casting_data/casting_data/test/def_front'




image_files_train_def = os.listdir(dir_train_def)
image_files_train_ok = os.listdir(dir_train_ok)

n = len(image_files_train_def)
m = len(image_files_train_ok)
print(f'the number of all the images in the training set is {n+m}')
print(f'number of def imgs is {n}')
print(f'number of ok imgs is {m}')
print(f'the ratio between ok and def imgs is {m/n}')


# Function to get a list of random image files from a directory
def get_random_image_files(directory, num_files):
    files = os.listdir(directory)
    random.shuffle(files)
    return files[:num_files]

# Create a 2x3 grid for "ok_front" images
plt.figure(figsize=(12, 8))
plt.suptitle('Examples From The Training Dataset')

for i in range(3):
    plt.subplot(2, 3, i + 1)
    image_files_ok = get_random_image_files(dir_train_ok, 3)
    img = Image.open(os.path.join(dir_train_ok, image_files_ok[i]))
    plt.imshow(img)
    plt.title('ok_front')

# Create a 2x3 grid for "def_front" images
for i in range(3):
    plt.subplot(2, 3, i + 4)
    image_files_def = get_random_image_files(dir_train_def, 3)
    img = Image.open(os.path.join(dir_train_def, image_files_def[i]))
    plt.imshow(img)
    plt.title('def_front')

plt.tight_layout()
plt.show()



img = Image.open(os.path.join(dir_train_def, image_files_def[0]))

print(img.size, img.mode)

# We can observe that we can generate more examples just using rotations
img_size = (300,300)
rand_seed = 555
batch_size = 32
epochs = 15

train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=40,   
    brightness_range=[0.2, 1.5], 
    validation_split=0.4,
)

test_gen = ImageDataGenerator(rescale=1./255)


arg_train = {'target_size': img_size,
             'color_mode': 'rgb',
             'classes': {'ok_front': 0,
                         'def_front': 1},
             'class_mode': 'binary',
             'batch_size': batch_size,
             'seed': rand_seed}

arg_test = {'target_size': img_size,
            'color_mode': 'rgb',
            'classes': {'ok_front': 0,
                        'def_front': 1},
            'class_mode': 'binary',
            'batch_size': batch_size,
            'seed': rand_seed,
            'shuffle': False}

dir_train='casting/casting_data/casting_data/train'
dir_test='casting/casting_data/casting_data/test'

# 80%
train_set = train_gen.flow_from_directory(directory=dir_train,
                                          subset='training',
                                          **arg_train)
#20%
valid_set = train_gen.flow_from_directory(directory=dir_train,
                                          subset='validation',
                                          **arg_train)

# for the 0 and 1 ...etc
test_set = test_gen.flow_from_directory(directory=dir_test,
                                        **arg_test)



model=Sequential()

model.add( Conv2D(filters=128,
                  kernel_size=(3,3),
                  strides=(2,2),
                  padding='valid',
                  kernel_initializer='he_uniform',
                  activation='relu',
                  input_shape=(300,300,3)
                  )
          )
model.add(MaxPooling2D(strides=(2,2),pool_size=(2,2),padding='valid'))

model.add(Conv2D(filters=128,
                  kernel_size=(3,3),
                  strides=(2,2),
                  padding='valid',
                  kernel_initializer='he_uniform',
                  activation='relu',
                  input_shape=(300,300,3)
                  )
)

model.add(MaxPooling2D(strides=(2,2),pool_size=(2,2),padding='valid'))

model.add(Flatten())
model.add(Dense(units=128,activation='relu',kernel_initializer='he_uniform'))
model.add(Dropout(0.2))
model.add(Dense(units=64,activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_set,validation_data=valid_set,epochs=10,verbose=1)

plt.subplot(1,2,1)
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Epochs')
plt.legend(['Train','Validation'],loc='upper left')

plt.subplot(1,2,2)
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Loss')
plt.ylabel('Epochs')
plt.legend(['Train','Validation'],loc='upper left')

model.save('modelByMe.h5')

newmodel=load_model('modelByMe.h5')

y_pred=newmodel.predict(test_set)
th=y_pred.mean()
y_pred=np.where(y_pred>th,1,0)

y_true=test_set.classes

from sklearn.metrics import confusion_matrix
 
import seaborn as sns
sns.heatmap( confusion_matrix(y_true,y_pred),annot=True,fmt='d')


