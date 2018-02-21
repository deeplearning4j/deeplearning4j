from .java_classes import *
import numpy as np

# Java instance initializations
native_ops = NativeOpsHolder.getInstance().getDeviceNativeOps()


# DATA TYPE MANAGEMENT

def set_context_dtype(dtype):
    '''
    Sets the dtype for nd4j
    # Arguments
        dtype: 'float' or 'double'
    '''
    dtype = DateTypeUtil.getDTypeFromContext(dtype)
    DateTypeUtil.setDTypeForContext(dtype)

def get_context_dtype():
    '''
    Returns the nd4j dtype
    '''
    dtype = DataTypeUtil.getDtypeFromContext()
    return DataTypeUtil.getDTypeForName(dtype)


def get_nd4j_dtype(np_dtype):
    '''
    Gets the equivalent nd4j data type
    for a given numpy data type.
    # Arguments
        np_dtype: Numpy data type. One of
            ['float64', 'float32', 'float16']
    '''
    if type(np_dtype) == type:
        np_dtype = np_dtype.__name__
    elif type(np_dtype) == np.dtype:
        np_dtype = np_dtype.name
    mapping = {
        'float64' : 'double',
        'float32' : 'float',
        'float16' : 'half'
        }
    nd4j_dtype = mapping.get(np_dtype)
    if not nd4j_dtype:
        raise Exception('Invalid numpy data type : ' + np_dtype)
    return nd4j_dtype

def get_np_dtype(nd4j_dtype):
    '''
    Gets the equivalent numpy data type
    for a given nd4j data type.
    # Arguments:
        nd4j_dtype : Nd4j data type. One of 
        ['double', 'float', 'half']
    '''
    mapping = {
        'double' : np.float64,
        'float' : np.float32,
        'half' : np.float16
    }
    np_dtype = mapping.get(nd4j_dtype)
    if not np_dtype:
        raise Exception('Invalid nd4j data type : ' + nd4j_dtype)
    return np_dtype

_refs = []

def from_numpy(np_array):
    '''
    Convert numpy array to nd4j array
    '''

    # Convert the numpy array to nd4j context dtype
    required_dtype = get_np_dtype(get_context_dtype())
    if np_array.dtype != required_dtype:
        np_array = np.cast[required_dtype](np_array)

    # Nd4j does not have 1-d vectors.
    # So we add a dummy dimension.
    if np_array.ndim == 1:
        np_array = np.expand_dims(np_array, 0)

    # We have to maintain references to all incoming
    # numpy arrays. Else they will get GCed
    _refs.append(np_array)

    # creates a Nd4j array from a numpy array
    # To create an Nd4j array, we need 3 things:
    # buffer, strides, and shape

    # Get the buffer
    # A buffer is basically an array. To get the buffer object
    # we need a pointer to the first element and the size.
    pointer_address, _ = np_array.__array_interface__['data']
    pointer = native_ops.pointerForAddress(pointer_address)
    size = np_array.size
    mapping = {
        np.float64 : DoublePointer,
        np.float32 : FloatPointer,
    }
    pointer = mapping[required_dtype](pointer)
    buff = Nd4j.createBuffer(pointer, size)

    # Get the strides
    # strides = tuple of bytes to step in each 
    # dimension when traversing an array.
    elem_size = buff.getElementSize()
    # Make sure word size is same in both python
    # and java worlds
    assert elem_size == np_array.dtype.itemsize
    strides = np_array.strides
    # numpy uses byte wise strides. We have to 
    # convert it to word wise strides.
    strides = [dim / elem_size for dim in strides]

    # Finally, shape:
    shape = np_array.shape

    nd4j_array = Nd4j.create(buff, shape, strides, 0)
    return nd4j_array


class ndarray(object):

    def __init__(self, data, dtype=None):
        # we ignore dtype for now
        typ = type(data)
        if typ is INDArray:
            # Note that we don't make a copy here
            self.array = data
        elif typ is ndarray:
            self.array = data.array.dup()
        else:
            if typ is not np.ndarray:
                data = np.array(data)
            self.is1d = data.ndim == 1
            self.array = from_numpy(data)

    def numpy(self):
        # TODO: Too expensive. Make it cheaper.
        array = self.array
        get = array.getDouble
        shape = array.shape()
        length = array.length()
        scalars = [get(i) for i in range(length)]
        return np.array(scalars).reshape(shape)

    @property
    def size(self):
        return self.array.length()

    @property
    def shape(self):
        return tuple(self.array.shape())

    @property
    def ndim(self):
        return len(self.array.shape())

    def __getitem__(self, key):
        if type(key) is int:
            array = ndarray(self.array.get(NDArrayIndex.point(key)))
            if self.is1d:
                array.is1d = True
            else:
                if self.array.shape()[0] == 1:
                    assert key in (0, -1), 'Index ' + str(key) +
                    'is out of bounds for axis 0 with size 1'
                    array.is1d = True
            return array
        if type(key) is slice:
            start = key.start
            stop = key.stop
            step = key.step
            if start is None:
                start = 0
            if stop is None:
                if self.is1d:
                    stop = self.array.size(1)
                else:
                    stop = self.array.size(0)
            if step is None or step == 1:
                idx = NDArrayIndex.interval(start, stop)
            else:
                idx = NDArrayIndex.interval(start, step, stop)
            array = self.array.get(idx)
            return ndarray(array)
        if type(key) in (list, tuple):
            key = list(key)
            args = []
            for i, k in enumerate(key):
                if type(dim) is int:
                    args.append(NDArrayIndex.point(dim))
                if type(dim) is slice:
                    pass




def array(*args, **kwargs):
    return ndarray(*args, **kwargs)