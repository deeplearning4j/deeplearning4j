import h5py
import tempfile
import new
import numpy
import xxhash

from keras import backend as K
from os import path, mkdir
from py4j.java_gateway import JavaGateway

from common import *

########
# for any Model/Sequential instances
########


def _save_model(
        self,
        filepath,
        overwrite=True,
        saveUpdaterState=False,
        useLegacySave=False):
    """
    Save model to disk in DL4J format.
    """

    if useLegacySave:
        self._old_save(filepath,overwrite)

    else:
        check_dl4j_model(self) # enforces dl4j model for model.fn()

        if self.__class__.__name__ == 'Sequential':
            params_builder = gateway.jvm.org.deeplearning4j.keras.api.SaveParams.builder()
            params_builder.sequentialModel(self._dl4j_model)
            params_builder.writePath(filepath)
            params_builder.saveUpdaterState(saveUpdaterState)
            gateway.sequentialSave(params_builder.build())

        elif self.__class__.__name__ == 'Model':
            params_builder = gateway.jvm.org.deeplearning4j.keras.api.SaveParams.builder()
            params_builder.functionalModel(self._dl4j_model)
            params_builder.writePath(filepath)
            params_builder.saveUpdaterState(saveUpdaterState)
            gateway.functionalSave(params_builder.build())

        else:
            raise ValueError('DL4J Keras only works with Sequential and Functional models')
