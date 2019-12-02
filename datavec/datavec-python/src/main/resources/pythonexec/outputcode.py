
def __is_numpy_array(x):
    return str(type(x))== "<class 'numpy.ndarray'>"

def maybe_serialize_ndarray_metadata(x):
    return serialize_ndarray_metadata(x) if __is_numpy_array(x) else x


def serialize_ndarray_metadata(x):
    return {"address": x.__array_interface__['data'][0],
            "shape": x.shape,
            "strides": x.strides,
            "dtype": str(x.dtype),
            "_is_numpy_array": True} if __is_numpy_array(x) else x


def is_json_ready(key, value):
    return key is not 'f2' and not inspect.ismodule(value) \
           and not hasattr(value, '__call__')

