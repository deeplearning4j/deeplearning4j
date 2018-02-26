import keras
from keras.models import Sequential, save_model
from keras.layers import SimpleRNN
import keras.backend as K

base_path = "../../../../../../../../resources/weights/"
backend = K.backend()
version = keras.__version__
major_version = int(version[0])

n_in = 4
n_out = 6

model = Sequential()
model.add(SimpleRNN(n_out, input_shape=(n_in, 1)))

model.compile(loss='mse', optimizer='adam')

print("Saving model with single simple RNN layer for backend {} and keras major version {}".format(backend, major_version))
model.save("{}simple_rnn_{}_{}.h5".format(base_path, backend, major_version))
