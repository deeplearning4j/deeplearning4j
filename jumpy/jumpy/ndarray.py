################################################################################
# Copyright (c) 2015-2018 Skymind, Inc.
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################


from .java_classes import *
import numpy as np
import ctypes


# Java instance initializations
native_ops = NativeOpsHolder.getInstance().getDeviceNativeOps()


# DATA TYPE MANAGEMENT

def set_context_dtype(dtype):
    '''
    Sets the dtype for nd4j
    # Arguments
        dtype: 'float' or 'double'
    '''
    dtype_map = {
        'float32': 'float',
        'float64': 'double'
    }
    dtype = dtype_map.get(dtype, dtype)
    if dtype not in ['float', 'double']:
        raise ValueError("Invalid dtype '{}'. Available dtypes are 'float' and 'double'.".format(dtype))
    dtype_ = DataTypeUtil.getDtypeFromContext(dtype)
    DataTypeUtil.setDTypeForContext(dtype_)
    if get_context_dtype() != dtype:
        raise RuntimeError("Can not set context dtype now. Set it at the beginning of your program.")


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
        'float64': 'double',
        'float32': 'float',
        'float16': 'half'
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
        'double': np.float64,
        'float': np.float32,
        'half': np.float16
    }
    np_dtype = mapping.get(nd4j_dtype)
    if not np_dtype:
        raise Exception('Invalid nd4j data type : ' + nd4j_dtype)
    return np_dtype


set_context_dtype('double')


_refs = []


def _from_numpy(np_array):
    '''
    Convert numpy array to nd4j array
    '''

    # Convert the numpy array to nd4j context dtype
    required_dtype = get_np_dtype(get_context_dtype())
    if np_array.dtype != required_dtype:
        raise Exception("{} is required. Got {} instead.".format(repr(required_dtype), np_array.dtype))

    # Nd4j does not have 1-d vectors.
    # So we add a dummy dimension.
    if np_array.ndim == 1:
        np_array = np.expand_dims(np_array, 0)

    # We have to maintain references to all incoming
    # numpy arrays. Else they will get GCed

    # creates a Nd4j array from a numpy array
    # To create an Nd4j array, we need 3 things:
    # buffer, strides, and shape

    # Get the buffer
    # A buffer is basically an array. To get the buffer object
    # we need a pointer to the first element and the size.
    pointer_address, _ = np_array.__array_interface__['data']
    _refs.append(np_array)
    pointer = native_ops.pointerForAddress(pointer_address)
    size = np_array.size
    mapping = {
        np.float64: DoublePointer,
        np.float32: FloatPointer,
    }
    pointer = mapping[required_dtype](pointer)
    buff = Nd4j.createBuffer(pointer, size)
    assert buff.address() == pointer_address
    _refs.append(buff)
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
    assert buff.address() == nd4j_array.data().address()
    return nd4j_array


def _to_numpy(nd4j_array):
    '''
    Convert nd4j array to numpy array
    '''
    buff = nd4j_array.data()
    address = buff.pointer().address()
    dtype = get_context_dtype()
    mapping = {
        'double': ctypes.c_double,
        'float': ctypes.c_float
    }
    Pointer = ctypes.POINTER(mapping[dtype])
    pointer = ctypes.cast(address, Pointer)
    np_array = np.ctypeslib.as_array(pointer, tuple(nd4j_array.shape()))
    return np_array


def _indarray(x):
    if type(x) is INDArray:
        return x
    elif type(x) is ndarray:
        return x.array
    elif 'numpy' in str(type(x)):
        return _from_numpy(x)
    elif type(x) in (list, tuple):
        return _from_numpy(np.array(x))
    elif type(x) in (int, float):
        return Nd4j.scalar(x)
    else:
        raise Exception('Data type not understood :' + str(type(x)))


def broadcast_like(y, x):
    xs = x.shape()
    ys = y.shape()
    if xs == ys:
        return y
    _xs = tuple(xs)
    _ys = tuple(ys)
    nx = len(xs)
    ny = len(ys)
    if nx > ny:
        diff = nx - ny
        ys = ([1] * diff) + ys
        y = y.reshape(ys)
        ny = nx
    elif ny > nx:
        raise Exception('Unable to broadcast shapes ' + str(_xs) + ''
                        ' and ' + str(_ys))
    yt = []
    rep_y = False
    for xd, yd in zip(xs, ys):
        if xd == yd:
            yt.append(1)
        elif xd == 1:
            raise Exception('Unable to broadcast shapes ' + str(_xs) + ''
                            ' and ' + str(_ys))
        elif yd == 1:
            yt.append(xd)
            rep_y = True
        else:
            raise Exception('Unable to broadcast shapes ' + str(_xs) + ''
                            ' and ' + str(_ys))
    if rep_y:
        y = y.repmat(*yt)
    return y


def broadcast(x, y):
    xs = x.shape()
    ys = y.shape()
    if xs == ys:
        return x, y
    _xs = tuple(xs)
    _ys = tuple(ys)
    nx = len(xs)
    ny = len(ys)
    if nx > ny:
        diff = nx - ny
        ys = ([1] * diff) + ys
        y = y.reshape(*ys)
        ny = nx
    elif ny > nx:
        diff = ny - nx
        xs = ([1] * diff) + xs
        x = x.reshape(*xs)
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
            raise Exception('Unable to broadcast shapes ' + str(_xs) + ''
                            ' and ' + str(_ys))
    if rep_x:
        x = Nd4j.tile(x, *xt)
    if rep_y:
        y = Nd4j.tile(y, *yt)
    return x, y


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
            self.array = _from_numpy(data)

    def numpy(self):
        # TODO: Too expensive. Make it cheaper.
        np_array = _to_numpy(self.array)
        return np_array

    @property
    def size(self):
        return self.array.length()

    @property
    def shape(self):
        return tuple(self.array.shape())

    @shape.setter
    def shape(self, value):
        arr = self.reshape(value)
        self.array = arr.array

    @property
    def ndim(self):
        return len(self.array.shape())


    def __getitem__(self, key):
        if type(key) is int:
            return ndarray(self.array.get(NDArrayIndex.point(key)))
        if type(key) is slice:
            start = key.start
            stop = key.stop
            step = key.step
            if start is None:
                start = 0
            if stop is None:
                shape = self.array.shape()
                if shape[0] == 1:
                    stop = shape[1]
                else:
                    stop = shape[0]
            if stop - start <= 0:
                return None
            if step is None or step == 1:
                return ndarray(self.array.get(NDArrayIndex.interval(start, stop)))
            else:
                return ndarray(self.array.get(NDArrayIndex.interval(start, step, stop)))
        if type(key) is list:
            raise NotImplementedError(
                'Sorry, this type of indexing is not supported yet.')
        if type(key) is tuple:
            key = list(key)
            shape = self.array.shape()
            ndim = len(shape)
            nk = len(key)
            key += [slice(None)] * (ndim - nk)
            args = []
            for i, dim in enumerate(key):
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
                            stop = shape[i]
                        if stop - start <= 0:
                            return None
                        if step is None or step == 1:
                            args.append(NDArrayIndex.interval(start, stop))
                        else:
                            args.append(NDArrayIndex.interval(
                                start, step, stop))
                elif type(dim) in (list, tuple):
                    raise NotImplementedError(
                        'Sorry, this type of indexing is not supported yet.')
            return ndarray(self.array.get(*args))

    def __setitem__(self, key, other):
        other = _indarray(other)
        view = self[key]
        if view is None:
            return
        view = view.array
        other = broadcast_like(other, view)
        view.assign(other)

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

    def __pow__(self, other):
        other = _indarray(other)
        x, y = broadcast(self.array, other)
        return ndarray(Transforms.pow(x, y))

    def __iadd__(self, other):
        other = _indarray(other)
        if self.array.shape() == other.shape():
            self.array = self.array.addi(other)
        else:
            x, y = broadcast(self.array, other)
            self.array = x.add(y)
        return self

    def __isub__(self, other):
        other = _indarray(other)
        if self.array.shape() == other.shape():
            self.array = self.array.subi(other)
        else:
            x, y = broadcast(self.array, other)
            self.array = x.sub(y)
        return self

    def __imul__(self, other):
        other = _indarray(other)
        if self.array.shape() == other.shape():
            self.array = self.array.muli(other)
        else:
            x, y = broadcast(self.array, other)
            self.array = x.mul(y)
        return self

    def __idiv__(self, other):
        other = _indarray(other)
        if self.array.shape() == other.shape():
            self.array = self.array.divi(other)
        else:
            x, y = broadcast(self.array, other)
            self.array = x.div(y)
        return self

    def __ipow__(self, other):
        other = _indarray(other)
        if self.array.shape() == other.shape():
            self.array = self.array.divi(other)
        else:
            x, y = broadcast(self.array, other)
            self.array = Transforms.pow(x, y)
        return self
    
    def __getattr__(self, attr):
        import ops
        f = getattr(ops, attr)
        setattr(ndarray, attr, f)
        return getattr(self, attr)

    def __int__(self):
        if self.array.length() == 1:
            return self.array.getInt(0)
        raise Exception('Applicable only for scalars')

    def __float__(self):
        if self.array.length() == 1:
            return self.array.getDouble(0)
        raise Exception('Applicable only for scalars')

    @property
    def T(self):
        return self.transpose()


def array(*args, **kwargs):
    return ndarray(*args, **kwargs)
