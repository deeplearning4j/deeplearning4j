import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Reshape
import keras.backend as K

base_path = "../../../../../../../../resources/weights/"
backend = K.backend()
version = keras.__version__
major_version = int(version[0])

if major_version == 2:
    from keras.layers import Conv2D
else:
    from keras.layers import Convolution2D as Conv2D



model = Sequential()


in_shape = (100,)
depth = 10
kernel_size = (3, 3)
h = 7
w = 7
n_out = 5

model.add(Dense(depth * h * w, input_shape=in_shape))
model.add(Reshape((h, w, depth))) # (7,7,10) - need to make sure DL4J detects 10 as channels
model.add(BatchNormalization())
if major_version == 2:
    model.add(Conv2D(n_out, kernel_size))
else:
    model.add(Conv2D(n_out, kernel_size[0], kernel_size[1]))


model.compile(loss='mse', optimizer='sgd')

print("Saving model with dense into batchnorm into conv2d layer for backend {} and keras major version {}".format(backend, major_version))
model.save("{}batch_to_conv2d_{}_{}.h5".format(base_path, backend, major_version))
