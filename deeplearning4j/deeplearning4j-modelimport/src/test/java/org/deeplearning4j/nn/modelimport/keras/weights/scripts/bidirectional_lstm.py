import keras
from keras.models import Sequential, save_model
from keras.layers import LSTM, Bidirectional
import keras.backend as K

base_path = "../../../../../../../../resources/weights/"
backend = K.backend()
version = keras.__version__
major_version = int(version[0])

n_in = 4
n_out = 6

model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                        input_shape=(n_in, 10)))

model.compile(loss='mse', optimizer='adam')

print("Saving model with single Bidirectional LSTM layer for backend {} and keras major version {}".format(backend, major_version))
model.save("{}bidirectional_lstm_{}_{}.h5".format(base_path, backend, major_version))