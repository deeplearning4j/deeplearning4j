from .op import op
from ..java_classes import *


# Linear algebra
# https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.linalg.html


@op
def dot(arr, other):
	return arr.mmul(other)


@op
def tensordot(arr1, arr2, axes=2):
	shape1 = arr1.shape()
	shape2 = arr2.shape()
	if type(axes) is int:
		axes = [shape1[axes:], shape2[:axes]]
	elif type(axes) in [list, tuple]:
		axes = list(axes)
		for i in range(2):
			if type(axes[i]) is int:
				axes[i] = [axes[i]]
	return Nd4j.tensorMmul(arr1, arr2, axes)
