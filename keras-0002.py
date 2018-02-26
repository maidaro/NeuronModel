from keras.models import Sequential
from keras.layers import Dense
from keras.utils import *
import numpy as np

# OR logical gate via Keras

# define input data
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# define output data
y_data = np.array([[0], [1], [1], [1]])

model = Sequential()
model.add(Dense(1, activation='sigmoid', input_dim=2))
model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
model.fit(x_data, y_data, epochs=10000, verbose = 0, batch_size=None)
print('Score:{}'.format(model.evaluate(x_data, y_data, batch_size=None)))
print('Recall:{}'.format(model.predict(x_data, batch_size=None)))
