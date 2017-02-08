import h5py
import tempfile
import new
import numpy
import xxhash

from keras import backend as K
from os import path, mkdir
from py4j.java_gateway import JavaGateway

from kerasdl4j.common import *


########
# hijacked functions in model class
########

########
# for Sequential(Model) instances
########

def _sequential_compile(
        self,
        optimizer,
        loss,
        metrics=None,
        sample_weight_mode=None,
        **kwargs):
    """
    Configures the learning process.
    """
    # first call the old compile() method
    self._old_compile(optimizer, loss, metrics, sample_weight_mode)

    # then convert to DL4J instance
    check_dl4j_model(self) # enforces dl4j model for model.fn()


def _sequential_fit(
        self,
        x,
        y,
        batch_size=32,
        nb_epoch=10,
        verbose=1,
        callbacks=[],
        validation_split=0.,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        **kwargs):
    """
    Executes fitting of the model by using DL4J as backend
    :param model: Model to use
    :param nb_epoch: Number of learning epochs
    :param features_directory: Directory with feature batch files
    :param labels_directory: Directory with label batch files
    :return:
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn()

    training_x = None
    training_y = None
    validation_x = None
    validation_y = None
    do_validation = True

    if validation_data:
        training_x = dump_ndarray(batch_size, x)
        training_y = dump_ndarray(batch_size, y)

        if len(validation_data) == 2:
            val_x, val_y = validation_data
            validation_x = dump_ndarray(batch_size, val_x)
            validation_y = dump_ndarray(batch_size, val_y)
        elif len(validation_data) == 3:
            val_x, val_y, val_sample_weight = validation_data
            validation_x = dump_ndarray(batch_size, val_x)
            validation_y = dump_ndarray(batch_size, val_y)
        else:
            raise ValueError('Incorrect configuration for validation_data. Must be a tuple of length 2 or 3.')
    elif validation_split and 0. < validation_split < 1.:
        split_at = int(len(x[0]) * (1. - validation_split))
        x, val_x = (slice_X(x, 0, split_at), slice_X(x, split_at))
        y, val_y = (slice_X(y, 0, split_at), slice_X(y, split_at))
        training_x = dump_ndarray(batch_size, x)
        training_y = dump_ndarray(batch_size, y)
        validation_x = dump_ndarray(batch_size, val_x)
        validation_y = dump_ndarray(batch_size, val_y)
    else:
        do_validation = False
        training_x = dump_ndarray(batch_size, x)
        training_y = dump_ndarray(batch_size, y)

    # gateway = JavaGateway()

    params_builder = gateway.jvm.org.deeplearning4j.keras.api.FitParams.builder()
    params_builder.sequentialModel(self._dl4j_model)
    params_builder.nbEpoch(nb_epoch)
    params_builder.trainXPath(training_x)
    params_builder.trainYPath(training_y)
    if not validation_x == None:
        params_builder.validationXPath(validation_x)
        params_builder.validationYPath(validation_y)
    params_builder.dimOrdering(K.image_dim_ordering())
    params_builder.doValidation(do_validation)
    gateway.sequentialFit(params_builder.build())


def _sequential_evaluate(
        self,
        x,
        y,
        batch_size=32,
        verbose=1,
        sample_weight=None,
        **kwargs):
    """
    Computes the loss on some input data, batch by batch.
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn()

    features_directory = dump_ndarray(batch_size, x)
    labels_directory = dump_ndarray(batch_size, y)

    # gateway = JavaGateway()

    params_builder = gateway.jvm.org.deeplearning4j.keras.api.EvaluateParams.builder()
    params_builder.sequentialModel(self._dl4j_model)
    params_builder.featuresDirectory(features_directory)
    params_builder.labelsDirectory(labels_directory)
    params_builder.batchSize(batch_size)
    gateway.sequentialEvaluate(params_builder.build())


def _sequential_predict(
        self,
        x,
        batch_size=32,
        verbose=0):
    """
    Generates output predictions for the input samples,
    processing the samples in a batched way.
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn()

    features_directory = dump_ndarray(batch_size, x)

    # gateway = JavaGateway()

    params_builder = gateway.jvm.org.deeplearning4j.keras.api.PredictParams.builder()
    params_builder.sequentialModel(self._dl4j_model)
    params_builder.featuresDirectory(features_directory)
    params_builder.batchSize(batch_size)
    gateway.sequentialPredict(params_builder.build())
    # TODO


def _sequential_predict_on_batch(
        self,
        x):
    """
    Returns predictions for a single batch of samples.
    """
    check_dl4j_model(self) # enforces dl4j model for model.fn()

    features_directory = dump_ndarray(len(x), x)

    # gateway = JavaGateway()

    params_builder = gateway.jvm.org.deeplearning4j.keras.api.PredictOnBatchParams.builder()
    params_builder.sequentialModel(self._dl4j_model)
    params_builder.featuresDirectory(features_directory)
    gateway.sequentialPredictOnBatch(params_builder.build())
    # TODO