from keras.datasets import mnist,cifar10
from matplotlib import pyplot as plt
import random
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout
import numpy as np
from keras.utils import to_categorical

import time


#plt.imshow(x_train[random.randint(0,59999)], cmap='gray')
#plt.show()

def fnn():
    #load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.reshape(x_train,(60000,784))
    xtest = np.reshape(x_test,(10000,784))
    y = to_categorical(y_train)
    ytest = to_categorical(y_test)

    model = Sequential()
    model.add(Dense(16,input_dim=784,activation = 'relu'))
    model.add(Dense(16,activation = 'relu'))
    model.add(Dense(10,activation = 'softmax'))
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.summary()
    start_time = time.time()
    history = model.fit(x, y, epochs=50, batch_size=800,verbose=2,validation_split = 0.05)
    results = model.evaluate(x = xtest,y = ytest,verbose = 2,batch_size = 200)
    print(time.time() - start_time)
    print(model.metrics_names,results)
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()


def cnnMNIST():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.reshape(x_train,(60000,28,28,1))
    xtest = np.reshape(x_test,(10000,28,28,1))
    y = to_categorical(y_train)
    ytest = to_categorical(y_test)

    model = Sequential()
    model.add(Conv2D(filters = 6, kernel_size = (3,3), strides=(1,1), data_format="channels_last", padding = 'same', input_shape = (28,28,1)))
    model.add(MaxPooling2D(pool_size = (2,2), padding = 'same', strides = 2))
    model.add(Conv2D(filters = 16, kernel_size = (3,3), strides=(1,1), data_format="channels_last", padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2), padding = 'same', strides = 2))
    model.add(Flatten())
    model.add(Dense(120,activation = 'relu'))
    model.add(Dense(84,activation = 'relu'))
    model.add(Dense(10,activation = 'softmax'))

    model.compile(optimizer='Adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.summary()
    model.save("model.h5")




    history = model.fit(x, y, epochs=10, batch_size=1024,verbose=2,validation_split = 0.05)
    print("done")
    results = model.evaluate(x = xtest,y = ytest,verbose = 1,batch_size = 32)
    print(model.metrics_names,results)
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    #json_string = model.to_json()
    #print(json_string)
    weights = model.layers[0].get_weights()
    print(weights)

def cnnCIFAR():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    xtrain = np.reshape(x_train,(50000,32,32,3))
    xtest = np.reshape(x_test,(10000,32,32,3))
    ytrain = to_categorical(y_train)
    ytest = to_categorical(y_test)
    plt.imshow(xtrain[random.randint(1,10000)], cmap='hsv')
    plt.show()

    model = Sequential()
    model.add(Conv2D(filters = 48, kernel_size = (3,3), strides=(1,1), data_format="channels_last", padding = 'same',activation = 'relu', input_shape = (32,32,3)))
    model.add(Conv2D(filters = 48, kernel_size = (3,3), strides=(1,1), data_format="channels_last", padding = 'same',activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2), padding = 'same', strides = 2))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 96, kernel_size = (3,3), strides=(1,1), data_format="channels_last", padding = 'same',activation = 'relu', input_shape = (32,32,3)))
    model.add(Conv2D(filters = 96, kernel_size = (3,3), strides=(1,1), data_format="channels_last", padding = 'same',activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2), padding = 'same', strides = 2))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 192, kernel_size = (3,3), strides=(1,1), data_format="channels_last", padding = 'same',activation = 'relu', input_shape = (32,32,3)))
    model.add(Conv2D(filters = 192, kernel_size = (3,3), strides=(1,1), data_format="channels_last", padding = 'same',activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2), padding = 'same', strides = 2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512,activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256,activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation = 'softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(xtrain, ytrain, epochs=1, batch_size=800,verbose=1,validation_split = 0.05)
    results = model.evaluate(x = xtest,y = ytest,verbose = 1,batch_size = 32)
    print(model.metrics_names,results)
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    model.save("cnn.h5")

cnnCIFAR()
