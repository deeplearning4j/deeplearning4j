from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Activation, Dense, Flatten

model = Sequential()

model.add(Conv2D(filters=1, kernel_size=(3, 3), input_shape=(10, 10, 3)))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(Activation('relu'))
model.add(Conv2D(filters=1, kernel_size=(3, 3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.save("cnn_batch_norm.h5")
