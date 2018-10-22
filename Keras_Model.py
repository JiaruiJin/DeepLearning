# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 10:12:13 2018
"""
from keras.models import Sequential 
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D 
from keras.layers import Flatten 
from keras.layers import Dense 
import numpy as np
import os
import glob
from skimage import io,transform
path='pathname'

# Set picture size
w=100
h=100
c=3

# read images from the file and lable them
def read_img(path_file):
    cate=[path+'/'+x for x in os.listdir(path) if os.path.isdir(path+'/'+x)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s'%(im))
            img=io.imread(im)
            img=transform.resize(img,(w,h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)

data,label=read_img(path)

# Divide pictures into random order for training and testing
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]

# Divide dataset to training and validation 
ratio=0.8
s=np.int(num_example*ratio)
x_train=data[:s]
y_train=label[:s]
x_val=data[s:]
y_val=label[s:]

# Initialising the CNN
model = Sequential()

# Convolution
model.add(Conv2D(32, (3, 3), input_shape = (100, 100, 3), activation = 'relu'))

# Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))

# Second convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
model.add(Flatten())

# Full connection
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs=10)
#test model accuracy
test_loss, test_acc = model.evaluate(x_val, y_val)
#save model
model.save('./Keras_Model.h5')
print('loss:', test_loss)
print('accuracy', test_acc)
