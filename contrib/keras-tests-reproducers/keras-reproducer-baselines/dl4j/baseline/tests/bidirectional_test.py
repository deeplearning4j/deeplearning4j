from keras import Sequential, Input
from keras.initializers.initializers import GlorotNormal
from keras.layers import Bidirectional, SimpleRNN
from keras.losses import MeanSquaredError
from keras.optimizers import Adam

from dl4j.baseline.model_manager import ModelManager

import numpy as np
import tensorflow as tf


class BidirectionalModelManager(ModelManager):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def set_default_inputs(self) -> None:
        inshape = (3, 10, 6)
        input_array = np.linspace(0, 1, num=np.prod(inshape), dtype=np.float32)
        input_array = input_array.reshape(inshape)
        in_ = tf.constant(input_array)
        self.set_inputs('model0', in_)
        self.set_inputs('model1', in_)
        self.set_inputs('model2', in_)

    def model_builder(self) -> None:
        # 'model0'
        model = Sequential()
        model.add(Input(shape=(10, 6)))
        model.add(Bidirectional(
            SimpleRNN(10, return_sequences=True, activation='tanh', kernel_initializer=GlorotNormal(),
                      recurrent_initializer=GlorotNormal())))
        model.compile(loss=MeanSquaredError(), optimizer=Adam())
        self.add_model('model0', model)

        # 'model1'
        model1 = Sequential()
        model1.add(Input(shape=(10, 6)))
        model1.add(SimpleRNN(10, return_sequences=True, activation='tanh', kernel_initializer=GlorotNormal(),
                            recurrent_initializer=GlorotNormal()))
        model1.compile(loss=MeanSquaredError(), optimizer=Adam())
        self.add_model('model1', model1)

        model3 = Sequential()
        model3.add(Input(shape=(10, 6)))
        model3.add(SimpleRNN(10, return_sequences=True, activation='tanh', kernel_initializer=GlorotNormal(),
                             recurrent_initializer=GlorotNormal()))
        model3.compile(loss=MeanSquaredError(), optimizer=Adam())
        self.add_model('model2', model3)
