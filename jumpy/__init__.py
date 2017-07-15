import jnius_config
import os
import inspect

import numpy as np
import ctypes

jnius_config.add_options('-Dorg.bytedeco.javacpp.nopointergc=true')

try:
    jnius_classpath = os.environ['JUMPY_CLASS_PATH']
except KeyError:
    raise 'Please specify a jar or directory for JUMPY_CLASS_PATH in the environment'


def get_classpath(base_path):
    """
    Get the classpath of based on the given folder.
    :param base_path: the directory to get the classpath for
    :return:
    """

    ret = ''
    for jar_file in os.listdir(base_path):
        ret += base_path + '/' + jar_file + ':'
    return ret


def _expand_directory(directory):
    if directory.__contains__('*'):
        # Get only the directory name (no wild card)
        jars = get_classpath(directory[:-2])
    else:
        jars = get_classpath(directory)

    return jars


new_class_path = ''
class_path_list = jnius_classpath.split(':')

if len(class_path_list) > 0 and len(class_path_list[0]) > 0:
    for class_path_item in class_path_list:
        if class_path_item.endswith('jar'):
            new_class_path += class_path_item
        # wild card expansions like /somedir/* (we do this because of jnius being unable to handle class path expansion
        else:
            new_class_path += _expand_directory(class_path_item)
            # update class path

else:
    class_path_item = jnius_classpath
    if class_path_item.endswith('jar'):
        new_class_path += class_path_item
        # wild card expansions like /somedir/* (we do this because of jnius being unable to handle class path expansion
    else:
        new_class_path += _expand_directory(class_path_item)

jnius_classpath = new_class_path

jnius_config.set_classpath(jnius_classpath)

# after jnius is initialized with proper class path *then* we setup nd4j

from jnius import autoclass

nd4j = autoclass('org.nd4j.linalg.factory.Nd4j')
INDArray = autoclass('org.nd4j.linalg.api.ndarray.INDArray')
transforms = autoclass('org.nd4j.linalg.ops.transforms.Transforms')
indexing = autoclass('org.nd4j.linalg.indexing.NDArrayIndex')

DataBuffer = autoclass('org.nd4j.linalg.api.buffer.DataBuffer')

system = autoclass('java.lang.System')
system.out.println(system.getProperty('org.bytedeco.javacpp.nopointergc'))
Integer = autoclass('java.lang.Integer')
Float = autoclass('java.lang.Float')
Double = autoclass('java.lang.Double')

nd4j_index = autoclass('org.nd4j.linalg.indexing.NDArrayIndex')

shape = autoclass('org.nd4j.linalg.api.shape.Shape')

serde = autoclass('org.nd4j.serde.binary.BinarySerde')

native_ops_holder = autoclass('org.nd4j.nativeblas.NativeOpsHolder')
native_ops = native_ops_holder.getInstance().getDeviceNativeOps()

DoublePointer = autoclass('org.bytedeco.javacpp.DoublePointer')
FloatPointer = autoclass('org.bytedeco.javacpp.FloatPointer')
IntPointer = autoclass('org.bytedeco.javacpp.IntPointer')

DataTypeUtil = autoclass('org.nd4j.linalg.api.buffer.util.DataTypeUtil')

MemoryManager = autoclass('org.nd4j.linalg.memory.MemoryManager')
memory_manager = nd4j.getMemoryManager()

import gc
gc.set_debug(gc.DEBUG_LEAK)

def disable_gc():
    memory_manager.togglePeriodicGc(False)


def set_gc_interval(interval=5000):
    memory_manager.setAutoGcWindow(interval)


def data_type():
    """
    Returns the data type name
    :return:
    """
    return DataTypeUtil.getDTypeForName(DataTypeUtil.getDtypeFromContext())


def set_data_type(data_type):
    """
    Set the data type for nd4j
    :param data_type: the data type to set
    one of:
    float
    double
    :return:
    """
    data_type_type = DataTypeUtil.getDtypeFromContext(data_type)
    DataTypeUtil.setDTypeForContext(data_type_type)


def dot(array1, array2):
    """
    The equivalent of numpy's "dot"
    :param array1: the first Nd4jArray
    :param array2: the second Nd4jArray
    :return: an nd4j array with the matrix multiplication
    result
    """
    return Nd4jArray(array1.array.mmul(array2.array))


def _get_numpy_buffer_reference(np_arr):
    return np.asarray(np_arr,dtype=_numpy_datatype_from_nd4j_context())


def get_buffer_from_arr(np_arr):
    """

    Create an nd4j data buffer from a numpy
    array's pointer

    :param np_arr: The input numpy array
    :return: and nd4j data buffer based
    on the numpy array's pointer
    """

    pointer_address = get_array_address(np_arr)
    pointer = native_ops.pointerForAddress(pointer_address)
    size = np_arr.size
    if np_arr.dtype == 'float64':
        as_double = DoublePointer(pointer)
        return Nd4jBuffer(nd4j.createBuffer(as_double, size))
    elif np_arr.dtype == 'float32':
        as_float = FloatPointer(pointer)
        return Nd4jBuffer(nd4j.createBuffer(as_float, size))
    elif np_arr.dtype == 'int64':
        as_int = IntPointer(pointer)
        return Nd4jBuffer(data_buffer=nd4j.createBuffer(as_int, size),
                          numpy_pointer=_get_numpy_buffer_reference(np_arr))


def _to_number(number):
    """
    Convert a number to a scalar ndarray
    :param number:
    :return:
    """
    return nd4j.scalar(number)


def get_array_address(np_arr):
    """
    :param np_arr: The numpy array to get the pointer address for
    :return:  the pointer address as a long
    """
    pointer, read_only_flag = np_arr.__array_interface__['data']
    return pointer


class Nd4jArray(object):
    """
     A small wrapper around nd4j's ndarray
     in java.
    """

    def __init__(self, nd4j_array=None,
                 numpy_array=None):
        self.array = nd4j_array
        self.numpy_array = numpy_array

    def __add__(self, other):
        if isinstance(other, Nd4jArray):
            return Nd4jArray(self.array.add(other.array), numpy_array=self.numpy_array)
        # scalar
        return Nd4jArray(nd4j_array=self.array.add(_to_number(other)), numpy_array=self.numpy_array)

    def __sub__(self, other):
        if isinstance(other, Nd4jArray):
            return Nd4jArray(nd4j_array=self.array.sub(other.array), numpy_array=self.numpy_array)
            # scalar
        return Nd4jArray(nd4j_array=self.array.sub(_to_number(other)), numpy_array=self.numpy_array)

    def __div__(self, other):
        if isinstance(other, Nd4jArray):
            return Nd4jArray(nd4j_array=self.array.div(other.array), numpy_array=self.numpy_array)
            # scalar
        return Nd4jArray(nd4j_array=self.array.div(_to_number(other)), numpy_array=self.numpy_array)

    def __mul__(self, other):
        if isinstance(other, Nd4jArray):
            return Nd4jArray(nd4j_array=self.array.mul(other.array), numpy_array=self.numpy_array)
            # scalar
        return Nd4jArray(nd4j_array=self.array.mul(_to_number(other)), numpy_array=self.numpy_array)

    def __gt__(self, other):
        if isinstance(other, Nd4jArray):
            return Nd4jArray(nd4j_array=self.array.gt(other.array), numpy_array=self.numpy_array)
            # scalar
        return Nd4jArray(nd4j_array=self.array.gt(_to_number(other)), numpy_array=self.numpy_array)

    def __lt__(self, other):
        if isinstance(other, Nd4jArray):
            return Nd4jArray(nd4j_array=self.array.lt(other.array), numpy_array=self.numpy_array)
            # scalar
        return Nd4jArray(nd4j_array=self.array.lt(_to_number(other)), numpy_array=self.numpy_array)

    def __deepcopy__(self, memodict={}):
        return Nd4jArray(nd4j_array=self.array.dup())

    def __eq__(self, other):
        if isinstance(other, Nd4jArray):
            return Nd4jArray(nd4j_array=self.array.add(other.array))
            # scalar
        return Nd4jArray(nd4j_array=self.array.add(_to_number(other)), numpy_array=self.numpy_array)

    def __imul__(self, other):
        if isinstance(other, Nd4jArray):
            return Nd4jArray(nd4j_array=self.array.muli(other.array), numpy_array=self.numpy_array)
            # scalar
        return Nd4jArray(nd4j_array=self.array.muli(_to_number(other)), numpy_array=self.numpy_array)

    def __isub__(self, other):
        if isinstance(other, Nd4jArray):
            return Nd4jArray(nd4j_array=self.array.subi(other.array), numpy_array=self.numpy_array)
            # scalar
        return Nd4jArray(nd4j_array=self.array.subi(_to_number(other)), numpy_array=self.numpy_array)

    def __iadd__(self, other):
        if isinstance(other, Nd4jArray):
            return Nd4jArray(nd4j_array=self.array.addi(other.array), numpy_array=self.numpy_array)
            # scalar
        return Nd4jArray(nd4j_array=self.array.addi(_to_number(other)), numpy_array=self.numpy_array)

    def __idiv__(self, other):
        if isinstance(other, Nd4jArray):
            return Nd4jArray(nd4j_array=self.array.divi(other.array))
            # scalar
        return Nd4jArray(nd4j_array=self.array.divi(_to_number(other)), numpy_array=self.numpy_array)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.array.getDouble(item)
        else:
            raise AssertionError("Only int types are supported for indexing right now")

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.array.putScalar(key, value)
        else:
            raise AssertionError("Only int types are supported for indexing right now")

    def rank(self):
        return self.array.rank()

    def length(self):
        return self.array.length()

    def shape(self):
        return self.array.shape()

    def stride(self):
        return self.array.stride()

    def data(self):
        return self.array.data()


methods = inspect.getmembers(INDArray, predicate=inspect.ismethod)
for name, method in methods:
    Nd4jArray.name = method


class Nd4jBuffer(object):
    def __init__(self, data_buffer=None, numpy_pointer=None):
        self.data_buffer = data_buffer
        self.numpy_pointer = numpy_pointer


    def __getitem__(self, item):
        if isinstance(item, int):
            return self.data_buffer.getDouble(item)
        else:
            raise AssertionError("Please ensure that item is of type int")

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.data_buffer.put(key, value)
        else:
            raise AssertionError("Please ensure that item is of type int")

    def length(self):
        return self.data_buffer.length()

    def element_size(self):
        return self.data_buffer.getElementSize()


methods = inspect.getmembers(DataBuffer, predicate=inspect.ismethod)
for name, method in methods:
    Nd4jBuffer.name = method


def _nd4j_datatype_from_np(np_datatype_name):
    """

    :param np_datatype_name:
    a numpy data type name.
    1 of:
    float64
    float32
    float16
    :return: the equivalent nd4j data type name (double,float,half)
    """
    if np_datatype_name == 'float64':
        return 'double'
    elif np_datatype_name == 'float32':
        return 'float'
    elif np_datatype_name == 'float16':
        return 'half'
    return None


def _nd4j_datatype_from_np_array(array):
    """
    Gets the equivalent nd4j datatype
    from the passed in numpy array

    :param array:
    :return:
    """
    return _nd4j_datatype_from_np(array.dtype.name)


def _numpy_datatype_from_nd4j_context():
    """
    Returns the appropriate
    numpy data type
    given the current nd4j context
    for data type
    :return:
    """
    nd4j_datatype = data_type()
    if nd4j_datatype == 'double':
        return np.float64
    elif nd4j_datatype == 'float':
        return np.float32
    elif nd4j_datatype == 'half':
        return np.float16


def _align_np_datatype_for_array(array):
    """
    Ensure the given numpy array
    matches the current nd4j data type
    :param array:
    :return:
    """
    return np.asarray(array, _numpy_datatype_from_nd4j_context())


def _assert_data_type_length(data_buffer):
    data_type = _numpy_datatype_from_nd4j_context()
    element_size = data_buffer.getElementSize()
    if data_type == np.float32 and element_size != 4:
        raise AssertionError("Data Type from nd4j is float. Data buffer size is not 4")
    elif data_type == np.float64 and element_size != 8:
        raise AssertionError("Data Type from nd4j is double. Data buffer size is not 8")
    elif data_type == np.int and element_size != 4:
        raise AssertionError("Data Type from nd4j is int. Data buffer size is not 4")


def from_np(np_arr):
    """

    Create an nd4j ndarray from a numpy array (passing the
    numpy pointer buffer by reference)

    :param np_arr: a numpy array
    :return:

    """

    np_arr = _align_np_datatype_for_array(np_arr)

    # nd4j doesn't have 1d arrays. Convert to a row vector
    if np_arr.ndim == 1:
        np_arr = np.reshape(np_arr, (1, np_arr.size))

    data_buffer = get_buffer_from_arr(np_arr).data_buffer
    _assert_data_type_length(data_buffer)
    #   note here we divide the strides by 8 for numpy
    # the reason we do this is because numpy's strides are based on bytes rather than words
    strides = map(lambda x: x / data_buffer.getElementSize(), np_arr.strides)
    arr_shape = np_arr.shape
    return Nd4jArray(nd4j_array=nd4j.create(data_buffer, arr_shape, strides, 0),
                     numpy_array=_get_numpy_buffer_reference(np_arr))
