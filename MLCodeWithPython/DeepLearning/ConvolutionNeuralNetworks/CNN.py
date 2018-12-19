#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 02:04:21 2018

@author: Venakta Kiran Menda
"""

# Convolution Neural Network

#Problem : Identifying image is Dog or Cat

# Data preprocessing is done manually bcoz no csv files but with jpg files

# part -1 Building a CNN

# Importing Keras libraries and packages


from keras.models import Sequential
from keras.layers import Convolution2D # prepraring Convolution layer
from keras.layers import MaxPooling2D# prepraring Pooling layer
from keras.layers import Flatten # prepraring Flattening 
from keras.layers import Dense# prepraring ANN layer

# Initialising the CNN

classifier = Sequential()


# CNN Building Steps
#---------------------
# Step 1 : Convloution
"""
32 - feture maps,
3,3 - 3*3 Matrix
input shape = 3 for 3 colors/ channels , 64,64 we are using since we have cpu tensor flow to reduce time 
(64,64, 3) for tensor flow
(3,64,64) for theano
activation = relu  which is rectifier function

"""

classifier.add(Convolution2D(32, 3, 3, input_shape = (64,64, 3), activation = 'relu'))

# Step2 : Pooling
# giving the stide 2 steps 
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Adding second convolution layer (first execute the code by commenting this code to check the results)
# keras know the shape so we remove input shape
classifier.add(Convolution2D(32, 3, 3,  activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 : Flattening

classifier.add(Flatten())

# Full Connection
# units taken by general consideration in the power of 2

classifier.add(Dense(units = 128, activation = 'relu')) # hidden layer we use relu
classifier.add(Dense(units = 1, activation = 'sigmoid')) # output layer we use sigmoid

# Compiling the CNN by using Stochastic Gradient with loss funciton 

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part-2 Fitting the CNN to the images
# we use keras documentation for image processing
# this code is taken form https://keras.io/preprocessing/image/ to use Image Augumentation

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory( 'dataset/training_set',
                                                target_size=(64, 64), # dimensions expected by CNN which we put in first step
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# fiting the model
classifier.fit_generator(training_set,
                        steps_per_epoch=8000, # training set total count
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000) # test set data count








