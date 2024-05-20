from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,GlobalAveragePooling2D,Conv2D,MaxPooling2D,ZeroPadding2D
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.models import Model

from sklearn.metrics import confusion_matrix,accuracy_score 
import numpy as np

from tensorflow.keras import backend as K

import tensorflow as tf
import matplotlib.pyplot as plt
import sys

#def addToModel(bottom_model,numclasses):



	#return top_model
	


img_height, img_width=100,100
batch_size=32
nb_epochs=5

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

print(input_shape)






# load the model
base_model = VGG16(weights='imagenet',include_top=False,input_shape=input_shape)

for layer in base_model.layers:
	layer.trainable=False




model= Sequential()
model.add(base_model) 
model.add(Flatten()) 



model.add(Dense(1024,activation=('relu'),input_dim=512))
model.add(Dense(512,activation=('relu'))) 
model.add(Dense(256,activation=('relu'))) 
model.add(Dropout(.3))
model.add(Dense(128,activation=('relu')))
#model.add(Dropout(.2))
model.add(Dense(1,activation=('sigmoid'))) 




print(model.summary())






train_data_dir= 'casting/casting_data/casting_data/'


print(train_data_dir)




#training stage

train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split


train_generator = train_datagen.flow_from_directory(
    train_data_dir + '/train/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir +'/train/', # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation') # set as validation data

test_gen = ImageDataGenerator(rescale=1./255)


            
test_set = test_gen.flow_from_directory(train_data_dir + '/test/',
                                        target_size=(img_height, img_width),
                                        batch_size=batch_size,
                                        class_mode='binary',
                                        )

    
model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print(model)



history=model.fit(
    train_generator,   
    validation_data = validation_generator,    
    epochs = nb_epochs)

model.save('v16.h5')



np.save('v16n.npy',history.history)




print(history.history)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(nb_epochs)

plt.figure(figsize=(8, 8))
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


newmodel=model.load_model('v16.h5')

y_pred=newmodel.predict(test_set)
th=y_pred.mean()
y_pred=np.where(y_pred>th,1,0)

y_true=test_set.classes

from sklearn.metrics import confusion_matrix
 
import seaborn as sns
sns.heatmap( confusion_matrix(y_true,y_pred),annot=True,fmt='d')










