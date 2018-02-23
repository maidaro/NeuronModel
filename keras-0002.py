import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import *

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=1000))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

data = np.random.random((1000, 1000))
labels = np.random.randint(10, size=(1000, 10))

#one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
model.fit(data, labels, epochs=10, batch_size=32)

plt.plot(data[:,0], labels[:,0], 'ro')
plt.legend()
plt.show()
