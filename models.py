import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

directory = r'/Users/paterosthodoris/Desktop/MSBA/MSBA_FALL_2023/STAT_6289/MIDTERM/dogscats/dogscats/train'
categories = ['cats','dogs']

img_size=100

data=[]

for category in categories:
	folder = os.path.join(directory,category)
	label = categories.index(category)
	for img in os.listdir(folder):
		img_path = os.path.join(folder, img)
		img_array = cv2.imread(img_path)
		img_array = cv2.resize(img_array, (img_size, img_size))
		data.append([img_array, label])

random.shuffle(data)

X=[]
y=[]

for features, labels in data:
	X.append(features)
	y.append(labels)

X=np.array(X)
y=np.array(y)

X=X/255

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
from keras.layers import Dropout
import time

NAME = f'cat-dog-pred-{int(time.time())}'
tensorboard=TensorBoard(log_dir=f'logs/{NAME}/')

model = Sequential()

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(512, input_shape=X.shape[1:], activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(512, input_shape=X.shape[1:], activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X,y,epochs=10,validation_split=0.1, callbacks=[tensorboard])