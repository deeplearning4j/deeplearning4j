import h5py
import tempfile
import new
import numpy
import xxhash

from keras import backend as K
from os import path, mkdir
from py4j.java_gateway import JavaGateway

from models import *
from functional import *
from sequential import *


def install_dl4j_backend(model):
    """
    Hijacks the `fit` method call in the model object. Detects
    if model is Sequential or Functional.
    :param model: Model in which fit will be hijacked
    """
    # append special methods
    model._old_save = model.save
    model.save = new.instancemethod(_save_model, model, None)

    # hijack Keras API
    if model.__class__.__name__ == 'Sequential':
        # compile()
        model._old_compile = model.compile
        model.compile = new.instancemethod(_sequential_compile, model, None)
        # fit()
        model._old_fit = model.fit
        model.fit = new.instancemethod(_sequential_fit, model, None)
        # evaluate()
        model._old_evaluate = model.evaluate
        model.evaluate = new.instancemethod(_sequential_evaluate, model, None)
        # predict()
        model._old_predict = model.predict
        model.predict = new.instancemethod(_sequential_predict, model, None)
        # predict_on_batch()
        model._old_predict_on_batch = model.predict_on_batch
        model.predict_on_batch = new.instancemethod(_sequential_predict_on_batch, model, None)

    elif model.__class__.__name__ == 'Model':
        # compile()
        model._old_compile = model.compile
        model.compile = new.instancemethod(_functional_compile, model, None)
        # fit()
        model._old_fit = model.fit
        model.fit = new.instancemethod(_functional_fit, model, None)
        # evaluate()
        model._old_evaluate = model.evaluate
        model.evaluate = new.instancemethod(_functional_evaluate, model, None)
        # predict()
        model._old_predict = model.predict
        model.predict = new.instancemethod(_functional_predict, model, None)
        # predict_on_batch()
        model._old_predict_on_batch = model.predict_on_batch
        model.predict_on_batch = new.instancemethod(_functional_predict_on_batch, model, None)

    else:
        raise ValueError('DL4J Keras only works with Sequential and Functional models')
