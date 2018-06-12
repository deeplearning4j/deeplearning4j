import keras
from keras.models import Sequential
import keras.backend as K
import numpy as np

base_path = "./"
backend = K.backend()
version = keras.__version__
major_version = int(version[0])

if major_version == 2:
    from keras.layers import Conv2D
else:
    from keras.layers import Convolution2D as Conv2D

input_shape=(5, 5, 5)
n_out = 6
kernel_size = (3, 3)

weights = np.arange(0, kernel_size[0] * kernel_size[1] * input_shape[0] * n_out)
weights = weights.reshape((kernel_size[0], kernel_size[1], input_shape[0], n_out))
bias = np.arange(0, n_out)

model = Sequential()
if major_version == 2:
    model.add(Conv2D(n_out, kernel_size, input_shape=input_shape))
else:
    model.add(Conv2D(n_out, kernel_size[0], kernel_size[1], input_shape=input_shape))

model.set_weights([weights, bias])

model.compile(loss='mse', optimizer='adam')

print("Saving model with single 2D convolution layer for backend {} and keras major version {}".format(backend, major_version))
model.save("{}conv2d_{}_{}.h5".format(base_path, backend, major_version))
