import keras
from keras.models import Sequential, save_model
from keras.layers import LSTM, Embedding
import keras.backend as K
import numpy as np

base_path = "../../../../../../../../resources/weights/"
backend = K.backend()
version = keras.__version__
major_version = int(version[0])

n_in = 4
n_out = 6
output_dim = 5
input_length = 10
mb = 42

model = Sequential()
model.add(Embedding(input_dim=n_in, output_dim=output_dim, input_length=input_length))
model.add(LSTM(n_out, return_sequences=True))

model.compile(loss='mse', optimizer='adam')

input_array = np.random.randint(n_in, size=(mb, input_length))

output_array = model.predict(input_array)
assert output_array.shape == (mb, input_length, n_out)

print("Saving model with embedding into LSTM layer for backend {} and keras major version {}".format(backend, major_version))
model.save("{}embedding_lstm_{}_{}.h5".format(base_path, backend, major_version))
