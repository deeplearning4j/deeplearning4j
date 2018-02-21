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

def _indarray(x):
    if type(x) is INDArray:
        return x
    elif type(x) is ndarray:
        return x.array
    elif 'numpy' in str(type(x)):
        return from_numpy(x)
    elif type(x) in (list, tuple):
        return from_numpy(np.array(x))
    elif type(x) in (int, float):
        return Nd4j.scalar(x)
    else:
        raise Exception('Data type not understood :' + str(type(x)))


def broadcast(x, y):
    xs = x.shape()
    ys = y.shape()
    _xs = tuple(xs)
    _ys = tuple(ys)
    if xs == ys:
        return x, y
    nx = len(xs)
    ny = len(ys)
    if nx > ny:
        diff = nx - ny
        ys += [1] * diff
        y = y.reshape(ys)
        ny = nx
    elif ny > nx:
        diff = ny - nx
        xs += [1] * diff
        x = x.reshape(xs)
        nx = ny
    xt = []
    yt = []
    rep_x = False
    rep_y = False
    for xd, yd in zip(xs, ys):
        if xd == yd:
            xt.append(1)
            yt.append(1)
        elif xd == 1:
            xt.append(yd)
            yt.append(1)
            rep_x = True
        elif yd == 1:
            xt.append(1)
            yt.append(xd)
            rep_y = True
        else:
            raise Exception('Unable to broadcast shapes ' + str(xs) + ''
                            ' and ' + str(ys))
    if rep_x:
        x = x.repmat(*xt)
    if rep_y:
        y = y.repmat(*yt)
    return x, y


class ndarray(object):

    def __init__(self, data, dtype=None):
        # we ignore dtype for now
        self.is0d = False
        self.is1d = False
        typ = type(data)
        if typ is INDArray:
            # Note that we don't make a copy here
            self.array = data
        elif typ is ndarray:
            self.array = data.array.dup()
            self.is0d = data.is0d
            self.is1d = data.is1d
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
        if self.is0d:
            return ()
        s = tuple(self.array.shape())
        if self.is1d:
            return s[1:]
        return s

    @property
    def ndim(self):
        if self.is0d:
            return 0
        if self.is1d:
            return 1
        return len(self.array.shape())

    @property
    def ndim(self):
        return len(self.array.shape())

    def __getitem__(self, key):
        if self.is0d:
            raise IndexError('Invalid index to scalar variable.')
        if type(key) is int:
            if self.is1d:
                array = ndarray(self.array.get(NDArrayIndex.point(key)))
                array.is0d = True
            else:
                if self.array.shape()[0] == 1:
                    assert key in (0, -1), 'Index ' + str(key) + ''
                    ' is out of bounds for axis 0 with size 1'
                    array = ndarray(self.array)
                    array.is1d = True
            return array
        if type(key) is slice:
            start = key.start
            stop = key.stop
            step = key.step
            if self.array.shape()[0] == 1 and not self.is1d:
                if start is None:
                    start = 0
                if stop is None:
                    stop = 1
                if stop - start > 0:
                    array = ndarray(self.array)
                    array.is1d = True
                    return array
                else:
                    return None ## We differ from numpy here.
                    # Instead of returning an empty array, 
                    # we return None.
            if key == slice(None):
                idx = NDArrayIndex.all()
            else:
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
            array = ndarray(array)
            array.is1d = self.is1d
            return array
        if type(key) is list:
            raise NotImplemented('Sorry, this type of indexing is not supported yet.')
        if type(key) is tuple:
            key = list(key)
            ndim = len(self.array.shape())
            ndim -= self.is1d
            nk = len(key)
            key += [slice(None)] * (ndim - nk)
            args = []
            set1d = False
            set0d = False
            if self.array.shape()[0] == 1:
                if self.is1d:
                    if type(key[0]) is int:
                        set0d = True
                    key.insert(0, 0)
                else:
                    zd = key[0]
                    if type(zd) is int:
                        assert zd in (0, -1), 'Index ' + str(zd) + ''
                        ' is out of bounds for axis 0 with size 1'
                        set1d = True
            for size, dim in enumerate(key):
                if type(dim) is int:
                    args.append(NDArrayIndex.point(dim))
                elif type(dim) is slice:
                    if dim == slice(None):
                        args.append(NDArrayIndex.all())
                    else:
                        start = dim.start
                        stop = dim.stop
                        step = dim.step
                        if start is None:
                            start = 0
                        if stop is None:
                            stop = size
                        if stop - start <= 0:
                            return None
                        if step is None or step == 1:
                            args.append(NDArrayIndex.interval(start, stop))
                        else:
                            args.append(NDArrayIndex.interval(start, step, stop))
                if type(dim) in (list, tuple):
                    raise NotImplemented('Sorry, this type of indexing is not supported yet.')
            array = ndarray(self.array.get(*args))
            if set0d:
                array.is0d = True
            elif set1d or self.is1d:
                array.is1d = True
            return array


    def __add__(self, other):
        other = _indarray(other)
        x, y = broadcast(self.array, other)
        return ndarray(x.add(y))

    def __sub__(self, other):
        other = _indarray(other)
        x, y = broadcast(self.array, other)
        return ndarray(x.sub(y))

    def __mul__(self, other):
        other = _indarray(other)
        x, y = broadcast(self.array, other)
        return ndarray(x.mul(y))

    def __div__(self, other):
        other = _indarray(other)
        x, y = broadcast(self.array, other)
        return ndarray(x.div(y))



def array(*args, **kwargs):
    return ndarray(*args, **kwargs)
