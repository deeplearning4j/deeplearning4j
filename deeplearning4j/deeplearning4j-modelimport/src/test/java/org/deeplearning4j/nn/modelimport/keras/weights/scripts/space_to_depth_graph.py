import keras
import keras.backend as K
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input, Dense, Lambda
import numpy as np

base_path = "./"
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



a_1 = Input(shape=(6, 6, 4))
a_2 = Input(shape=(3, 3, 16))

inputs = [a_1, a_2]

# TODO: simple space_to_batch test (using sequential)
# TODO: merging two conv type layers

std = Lambda(
    space_to_depth_x2,
    output_shape=space_to_depth_x2_output_shape,
    name='space_to_depth_x2')(a_1)

out = concatenate([std, a_2])
model = Model(inputs=inputs, outputs=out)

model.compile(loss='mse', optimizer='adam')

in_1 = np.random.rand(10, 6, 6, 4)
in_2 = np.random.rand(10, 3, 3, 16)

output = model.predict([in_1, in_2])

assert output.shape == (10, 3, 3, 16+16)  # concat

print("Saving model with space to depth layer merged with other inputs for backend {} and keras major version {}".format(backend, major_version))
model.save("{}space_to_depth_graph_{}_{}.h5".format(base_path, backend, major_version))
