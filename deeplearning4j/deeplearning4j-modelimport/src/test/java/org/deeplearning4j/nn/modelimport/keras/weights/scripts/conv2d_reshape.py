import keras
from keras.models import Sequential
import keras.backend as K
import numpy as np

base_path = "../../../../../../../../resources/weights/"
backend = K.backend()
version = keras.__version__
major_version = int(version[0])

if major_version == 2:
    from keras.layers import Conv2D
else:
    from keras.layers import Convolution2D as Conv2D
from keras.layers import Reshape, Dense, Flatten

input_shape=(5, 5, 5)
n_out = 6
kernel_size = (3, 2)

model = Sequential()
if major_version == 2:
    model.add(Conv2D(n_out, kernel_size, input_shape=input_shape))
else:
    model.add(Conv2D(n_out, kernel_size[0], kernel_size[1], input_shape=input_shape))

if backend == "tensorflow":
    model.add(Reshape((4, 3, n_out)))
else:
    model.add(Reshape((n_out, 4, 3)))

if major_version == 2:
    model.add(Conv2D(n_out, (1, 1)))
else:
    model.add(Conv2D(n_out, 1, 1))

model.add(Flatten())
model.add(Dense(12))

model.compile(loss='mse', optimizer='adam')

X = np.zeros((10, 5, 5, 5))
out = model.predict(X)

assert out.shape == (10, 12)

print("Saving model that reshapes conv2d output for backend {} and keras major version {}".format(backend, major_version))
model.save("{}conv2d_reshape_{}_{}.h5".format(base_path, backend, major_version))
