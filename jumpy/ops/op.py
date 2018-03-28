from ..java_classes import *
from ..ndarray import array, ndarray


_INDArray_class = 'org.nd4j.linalg.api.ndarray.INDArray'


def _is_nd4j(x):
	return type(x).__name__ == _INDArray_class


def _is_jumpy(x):
	return type(x) == ndarray

'''
Use the @op decorator over a method to automatically
take care of nd4j<->jumpy conversions. e.g:

```python

@op
def reshape(arr, shape):
	# we are in nd4j space now
	# arr is an INDArray instance
	# we return a INDArray instance as well
	return arr.reshape(*shape)


# use in jumpy space:

x = jp.zeros((2, 2, 3))  # x is jumpy ndarray
y = reshape(x, (4, 3))  # y is a jumpy ndarray

```

Note that methods with first argument named 'arr'
will be automatically bound to ndarray class.

'''
def op(f):
	def wrapper(*args, **kwargs):
		args = list(args)
		for i, arg in enumerate(args):
			if _is_jumpy(arg):
				args[i] = arg.array
		for k in kwargs:
			v = kwargs[k]
			if _is_jumpy(v):
				kwargs[k] = v.array
		out = f(*args, **kwargs)
		if _is_nd4j(out):
			return array(out)
		elif type(out) is list:
			for i, v in enumerate(out):
				if _is_nd4j(v):
					out[i] = array(v)
			return out
		elif type(out) is tuple:
			out = list(out)
			for i, v in enumerate(out):
				if _is_nd4j(v):
					out[i] = array(v)
			return tuple(out)
	return wrapper