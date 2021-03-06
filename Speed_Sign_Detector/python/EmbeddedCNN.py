__author__ = 'joseph'
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np

isInit = False
model = None

def initModel():
    global model
    global isInit

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

    model.load_weights('/home/joseph/PycharmProjects/CNN/CNN30ep_30k.h5')

    isInit = True

def RunPyCNN(img):
    global model
    global isInit    
    if(not isInit):
        initModel()
    
    ray = np.array(img)
    ray = ray.reshape((28,28))
    
    ray = ray / 255.0    
    CNN_image = ray.reshape((1,1, 28, 28))
    y = model.predict(CNN_image)     
    return y.flatten().tolist()


