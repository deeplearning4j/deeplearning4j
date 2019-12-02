def __is_numpy_array(x):
    return str(type(x))== "<class 'numpy.ndarray'>"

def __maybe_serialize_ndarray_metadata(x):
    return __serialize_ndarray_metadata(x) if __is_numpy_array(x) else x


def __serialize_ndarray_metadata(x):
    return {"address": x.__array_interface__['data'][0],
            "shape": x.shape,
            "strides": x.strides,
            "dtype": str(x.dtype),
            "_is_numpy_array": True} if __is_numpy_array(x) else x


def __serialize_list(x):
    import json
    return json.dumps(__recursive_serialize_list(x))


def __serialize_dict(x):
    import json
    return json.dumps(__recursive_serialize_dict(x))

def __recursive_serialize_list(x):
    out = []
    for i in x:
        if __is_numpy_array(i):
            out.append(__serialize_ndarray_metadata(i))
        elif isinstance(i, (list, tuple)):
            out.append(__recursive_serialize_list(i))
        elif isinstance(i, dict):
            out.append(__recursive_serialize_dict(i))
        else:
            out.append(i)
    return out

def __recursive_serialize_dict(x):
    out = {}
    for k in x:
        v = x[k]
        if __is_numpy_array(v):
            out[k] = __serialize_ndarray_metadata(v)
        elif isinstance(v, (list, tuple)):
            out[k] = __recursive_serialize_list(v)
        elif isinstance(v, dict):
            out[k] = __recursive_serialize_dict(v)
        else:
            out[k] = v
    return out