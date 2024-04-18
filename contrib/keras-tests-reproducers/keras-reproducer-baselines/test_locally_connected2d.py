import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model

for global_dtype in [tf.float64]:
    tf.keras.backend.set_floatx(global_dtype.name)

    for network_dtype in [tf.float64, tf.float32, tf.float16]:
        assert tf.keras.backend.floatx() == global_dtype.name

        for test in range(1,2):
            msg = f"Global dtype: {global_dtype}, network dtype: {network_dtype}, test={test}"

            if test == 0:
                inputs = keras.Input(shape=(4, 5))
                x = keras.layers.LSTM(5, return_sequences=True, dtype=network_dtype)(inputs)
                x = keras.layers.LocallyConnected1D(4, 2, dtype=network_dtype)(x)
                outputs = keras.layers.TimeDistributed(keras.layers.Dense(10, dtype=network_dtype))(x)
                model = keras.Model(inputs=inputs, outputs=outputs)

                in_data = tf.random.normal((2, 4, 5), dtype=network_dtype)
                label = tf.one_hot(tf.random.uniform((2, 4), maxval=10, dtype=tf.int32), depth=10)
                label = tf.cast(label, network_dtype)

            elif test == 1:
                inputs = keras.Input(shape=(8, 8, 1))
                x = keras.layers.Conv2D(5, 2, padding='same', dtype=network_dtype)(inputs)
                x = keras.layers.LocallyConnected2D(5, (2, 2), dtype=network_dtype)(x)
                outputs = keras.layers.Flatten()(x)
                outputs = keras.layers.Dense(10, dtype=network_dtype)(outputs)
                model = keras.Model(inputs=inputs, outputs=outputs)

                in_data = tf.random.normal((2, 8, 8, 1), dtype=network_dtype)
                label = tf.one_hot(tf.random.uniform((2,), maxval=10, dtype=tf.int32), depth=10)
                label = tf.cast(label, network_dtype)

            else:
                raise ValueError("Invalid test case")

            model.compile(optimizer='adam', loss='categorical_crossentropy')

            out = model(in_data)
            assert out.dtype == network_dtype, msg

            ff = model.predict(in_data)
            assert ff.dtype == network_dtype, msg

            model.fit(in_data, label, epochs=1, batch_size=2)