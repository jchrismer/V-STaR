import cPickle as pickle
import LoadData
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers
import cv2
import os

import EmbeddedCNN as E_CNN
import time

layout.append( ('conv2D', {'nb_filter': 100, 'nb_row': 7, 'nb_col': 7, 'init': 'uniform', 'activation': 'tanh', 'input_shape': input_shape}) )
	layout.append( ('maxpool2D', {'pool_size': (2,2)}) )
	layout.append( ('conv2D', {'nb_filter': 150, 'nb_row': 4, 'nb_col': 4, 'init': 'uniform', 'activation': 'tanh'}) )
	layout.append( ('maxpool2D', {'pool_size': (2,2)}) )
	layout.append( ('conv2D', {'nb_filter': 250, 'nb_row': 4, 'nb_col': 4, 'init': 'uniform', 'activation': 'tanh'}) )
	layout.append( ('maxpool2D', {'pool_size': (2,2)}) )
	layout.append( ('flatten', {}) )
	layout.append( ('dense', {'output_dim': 300, 'init': 'uniform', 'activation': 'tanh'}) )
	layout.append( ('dense', {'output_dim': num_classes, 'init': 'uniform', 'activation': 'softmax'}) )
	layouts.append( ('gtsrb', get_gtsrb_layout, {'input_shape': (3, 48, 48), 'num_classes': 43}) )

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(1, 28, 28)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(11))             #output layer
model.add(Activation('sigmoid'))


momentum = 0
nesterov = False
sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=momentum, nesterov=nesterov) # TODO parameters
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-2, momentum=0.9),
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        r'/home/joseph/Desktop/pi_car_proj/Main_Project_Folders/Database/Training/',  # leading r is recursive flag
        target_size=(28, 28),  # all images will be resized to 150x150
        batch_size=32,
        color_mode='grayscale',
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        r'/home/joseph/Desktop/pi_car_proj/Main_Project_Folders/Database/Evaluation/',
        target_size=(28, 28),
        batch_size=32,
        color_mode='grayscale',
        class_mode='categorical')

model.fit_generator(
        train_generator,
        samples_per_epoch=18331,        #18331
        nb_epoch=10,
        validation_data=validation_generator,
        nb_val_samples=1020)


model.save_weights('CNN_GTSRB.h5')  # always save your weights after training or during training

# Evaluate
'''
model.load_weights('CNN30ep_30k.h5')
'''
