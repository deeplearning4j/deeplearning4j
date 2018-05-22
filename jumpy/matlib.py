from .ndarray import ndarray
from .java_classes import Nd4j


def zeros(shape):
    return ndarray(Nd4j.zeros(*shape))


def ones(shape):
    return ndarray(Nd4j.ones(*shape))
