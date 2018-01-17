import keras
from keras.models import Sequential, save_model
from keras.layers import Dense
import keras.backend as K
import numpy as np

base_path = "../../../../../../../../resources/weights/"
backend = K.backend()
version = keras.__version__
major_version = int(version[0])

n_in = 4
n_out = 6

weights = np.arange(0, n_in * n_out).reshape((n_in, n_out))
bias = np.arange(0, n_out)

model = Sequential()
model.add(Dense(n_out, input_shape=(n_in,)))

model.set_weights([weights, bias])

model.compile(loss='mse', optimizer='adam')

print("Saving model with single dense layer for backend {} and keras major version {}".format(backend, major_version))
model.save("{}dense_{}_{}.h5".format(base_path, backend, major_version))
