import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Lambda
import numpy as np

base_path = "../../../../../../../../resources/weights/"
backend = K.backend()
version = keras.__version__
major_version = int(version[0])

def space_to_depth_x2(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=2)


def space_to_depth_x2_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2.
    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    4 * input_shape[3])


model = Sequential()
model.add(Lambda(
    space_to_depth_x2,
    output_shape=space_to_depth_x2_output_shape,
    name='space_to_depth_x2',
    input_shape=(6,6,4)))

model.compile(loss='mse', optimizer='adam')

input = np.random.rand(10, 6, 6, 4)

output = model.predict(input)
assert output.shape == (10, 3, 3, 16)


print("Saving model with a single space to depth layer for backend {} and keras major version {}".format(backend, major_version))
model.save("{}space_to_depth_simple_{}_{}.h5".format(base_path, backend, major_version))
