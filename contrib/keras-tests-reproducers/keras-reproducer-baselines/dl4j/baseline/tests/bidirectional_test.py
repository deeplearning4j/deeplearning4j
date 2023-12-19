from tensorflow.python.keras.layers.recurrent import SimpleRNN
from tensorflow.python.keras.layers.wrappers import Bidirectional
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.ops.init_ops import GlorotNormal

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
        in_ = tf.constant(input_array.reshape(inshape))


    def model_builder(self) -> None:
            # 'model0'
            model = Sequential()
            model.add(Bidirectional(
                SimpleRNN(10, return_sequences=True, activation='tanh', kernel_initializer=GlorotNormal(),
                          recurrent_initializer=GlorotNormal())))
            model.compile(loss='mean_squared_error', optimizer=Adam())
            self.add_model('model0', model)

            # 'model1'
            model = Sequential()
            model.add(SimpleRNN(10, return_sequences=True, activation='tanh', kernel_initializer=GlorotNormal(),
                                recurrent_initializer=GlorotNormal()))
            model.compile(loss='mean_squared_error', optimizer=Adam())
            self.add_model('model1', model)

            model3 = Sequential()
            model3.add(SimpleRNN(10, return_sequences=True, activation='tanh', kernel_initializer=GlorotNormal(),
                                 recurrent_initializer=GlorotNormal()))
            model3.compile(loss='mean_squared_error', optimizer=Adam())
