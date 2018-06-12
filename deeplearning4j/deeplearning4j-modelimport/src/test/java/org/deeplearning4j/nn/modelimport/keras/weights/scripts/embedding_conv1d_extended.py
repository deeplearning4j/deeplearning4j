import keras
from keras.models import Sequential, save_model
from keras.layers import Embedding, Convolution1D, Flatten, Dense, Dropout
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
kernel = 3

embedding_dim = 50
max_words = 200
input_length = 10

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=input_length))
model.add(Convolution1D(128, kernel_size=3, activation='relu')) # 10 - 3 + 1 = 8
model.add(Convolution1D(64, kernel_size=3, activation='relu')) # 10 - 3 + 1 = 6
model.add(Convolution1D(32, kernel_size=3, activation='relu')) # 10 - 3 + 1 = 4
model.add(Flatten()) # 128 = 32 * 4
model.add(Dropout(0.2))
model.add(Dense(128, activation='sigmoid')) # W = 128 x 128
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='mse', optimizer='adam')

input_array = np.random.randint(n_in, size=(mb, input_length))

output_array = model.predict(input_array)
assert output_array.shape == (mb, 1)

print("Saving model with embedding into several Conv1D layers into Flatten and Dense for backend {} and keras major version {}".format(backend, major_version))
model.save("{}embedding_conv1d_extended_{}_{}.h5".format(base_path, backend, major_version))
