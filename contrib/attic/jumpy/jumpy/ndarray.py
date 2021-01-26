#  /* ******************************************************************************
#   * Copyright (c) 2021 Deeplearning4j Contributors
#   *
#   * This program and the accompanying materials are made available under the
#   * terms of the Apache License, Version 2.0 which is available at
#   * https://www.apache.org/licenses/LICENSE-2.0.
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#   * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#   * License for the specific language governing permissions and limitations
#   * under the License.
#   *
#   * SPDX-License-Identifier: Apache-2.0
#   ******************************************************************************/

################################################################################
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
import warnings


native_ops = NativeOpsHolder.getInstance().getDeviceNativeOps()


# DATA TYPE MANAGEMENT


DOUBLE = DataType.DOUBLE
FLOAT = DataType.FLOAT
HALF = DataType.HALF
LONG = DataType.LONG
INT = DataType.INT
SHORT = DataType.SHORT
UBYTE = DataType.UBYTE
BYTE = DataType.BYTE
BOOL = DataType.BOOL
UTF8 = DataType.UTF8
COMPRESSED = DataType.COMPRESSED
UNKNOWN = DataType.UNKNOWN

SUPPORTED_JAVA_DTYPES = [
    DOUBLE,
    FLOAT,
    HALF,

    LONG,
    INT,
    SHORT,

    BOOL
    #UTF8
]

SUPPORTED_PYTHON_DTYPES = [
    np.float64,
    np.float32,
    np.float16,

    np.int64,
    np.int32,
    np.int16,

    np.bool_
    #np.str_
]




_PY2J = {SUPPORTED_PYTHON_DTYPES[i] : SUPPORTED_JAVA_DTYPES[i] for i in range(len(SUPPORTED_JAVA_DTYPES))}
_J2PY = {SUPPORTED_JAVA_DTYPES[i] : SUPPORTED_PYTHON_DTYPES[i] for i in range(len(SUPPORTED_JAVA_DTYPES))}


def _dtype_py2j(dtype):
    if isinstance(dtype, str):
        dtype = np.dtype(dtype).type
    elif isinstance(dtype, np.dtype):
        dtype = dtype.type
    jtype = _PY2J.get(dtype)
    if jtype is None:
        raise NotImplementedError("Unsupported type: " + dtype.name)
    return jtype


def _dtype_j2py(dtype):
    pytype = _J2PY.get(dtype)
    if pytype is None:
        raise NotImplementedError("Unsupported type: " + (str(dtype)))
    return pytype


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
        warnings.warn("Can not set context dtype now. Set it at the beginning of your program.")


def get_context_dtype():
    '''
    Returns the nd4j dtype
    '''
    dtype = DataTypeUtil.getDtypeFromContext()
    return DataTypeUtil.getDTypeForName(dtype)

_refs = []


def _from_numpy(np_array):
    '''
    Convert numpy array to nd4j array
    '''
    pointer_address, _ = np_array.__array_interface__['data']
    _refs.append(np_array)
    pointer = native_ops.pointerForAddress(pointer_address)
    size = np_array.size
    pointer.limit(size)
    jdtype = _dtype_py2j(np_array.dtype)
    '''
    mapping = {
        DOUBLE: DoublePointer,
        FLOAT: FloatPointer,
        HALF: HalfPointer,
        LONG: LongPointer,
        INT: IntPointer,
        SHORT: ShortPointer,
        BOOL: BoolPointer
        }
    pc = mapping[jdtype]
    #pointer = pc(pointer)
    '''
    buff = Nd4j.createBuffer(pointer, size, jdtype)
    assert buff.address() == pointer_address
    _refs.append(buff)
    elem_size = buff.getElementSize()
    assert elem_size == np_array.dtype.itemsize
    strides = np_array.strides
    strides = [dim / elem_size for dim in strides]
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
    dtype = nd4j_array.dataType().toString()
    mapping = {
        'DOUBLE': ctypes.c_double,
        'FLOAT': ctypes.c_float,
        'HALF': ctypes.c_short,
        'LONG': ctypes.c_long,
        'INT': ctypes.c_int,
        'SHORT': ctypes.c_short,
        'BOOL': ctypes.c_bool
    }
    Pointer = ctypes.POINTER(mapping[dtype])
    pointer = ctypes.cast(address, Pointer)
    np_array = np.ctypeslib.as_array(pointer, tuple(nd4j_array.shape()))
    return np_array


def _indarray(x):
    typ = type(x)
    if typ is INDArray:
        return x
    elif typ is ndarray:
        return x.array
    elif 'numpy' in str(typ):
        return _from_numpy(x)
    elif typ in (list, tuple):
        return _from_numpy(np.array(x))
    elif typ in (int, float):
        return Nd4j.scalar(x)
    else:
        raise Exception('Data type not understood :' + str(typ))


def _nparray(x):
    typ = type(x)
    if typ is INDArray:
        return ndarray(x).numpy()
    elif typ is ndarray:
        return x.numpy()
    elif 'numpy' in str(typ):
        return x
    elif typ in (list, tuple):
        return np.array(x)
    elif typ in (int, float):
        return np.array(x)
    else:
        raise Exception('Data type not understood :' + str(typ))


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
        try:
            y = Nd4j.tile(y, *yt)
        except:
            y = Nd4j.tile(y, *yt)
    return x, y


class ndarray(object):

    def __init__(self, data, dtype=None):
        # we ignore dtype for now
        typ = type(data)
        if 'nd4j' in typ.__name__:
            # Note that we don't make a copy here
            self.array = data
        elif typ is ndarray:
            self.array = data.array.dup()
        else:
            if typ is not np.ndarray:
                data = np.array(data)
            self.array = _from_numpy(data)

    def numpy(self):
        try:
            return self.np_array
        except AttributeError:
            self.np_array = _to_numpy(self.array)
            return self.np_array

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
        return ndarray(self.numpy()[key])
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
        self.numpy()[key] = _nparray(other)
        return
        other = _indarray(other)
        view = self[key]
        if view is None:
            return
        view = view.array
        other = broadcast_like(other, view)
        view.assign(other)

    def __add__(self, other):
        return ndarray(self.numpy() + _nparray(other))
        other = _indarray(other)
        x, y = broadcast(self.array, other)
        return ndarray(x.add(y))

    def __sub__(self, other):
        return ndarray(self.numpy() - _nparray(other))
        other = _indarray(other)
        x, y = broadcast(self.array, other)
        return ndarray(x.sub(y))

    def __mul__(self, other):
        return ndarray(self.numpy() * _nparray(other))
        other = _indarray(other)
        x, y = broadcast(self.array, other)
        return ndarray(x.mul(y))

    def __div__(self, other):
        return ndarray(self.numpy() / _nparray(other))
        other = _indarray(other)
        x, y = broadcast(self.array, other)
        return ndarray(x.div(y))

    def __pow__(self, other):
        return ndarray(self.numpy() ** _nparray(other))
        other = _indarray(other)
        x, y = broadcast(self.array, other)
        return ndarray(Transforms.pow(x, y))

    def __iadd__(self, other):
        self.numpy().__iadd__(_nparray(other))
        return self
        other = _indarray(other)
        if self.array.shape() == other.shape():
            self.array = self.array.addi(other)
        else:
            x, y = broadcast(self.array, other)
            self.array = x.add(y)
        return self

    def __isub__(self, other):
        self.numpy().__isub__(_nparray(other))
        return self
        other = _indarray(other)
        if self.array.shape() == other.shape():
            self.array = self.array.subi(other)
        else:
            x, y = broadcast(self.array, other)
            self.array = x.sub(y)
        return self

    def __imul__(self, other):
        self.numpy().__imul__(_nparray(other))
        return self
        other = _indarray(other)
        if self.array.shape() == other.shape():
            self.array = self.array.muli(other)
        else:
            x, y = broadcast(self.array, other)
            self.array = x.mul(y)
        return self

    def __idiv__(self, other):
        self.numpy().__idiv__(_nparray(other))
        return self
        other = _indarray(other)
        if self.array.shape() == other.shape():
            self.array = self.array.divi(other)
        else:
            x, y = broadcast(self.array, other)
            self.array = x.div(y)
        return self

    def __ipow__(self, other):
        self.numpy().__ipow__(_nparray(other))
        return self
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
