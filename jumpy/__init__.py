import jnius_config
import os

try:
    jnius_classpath = os.environ['JUMPY_CLASS_PATH']
except KeyError:
    raise 'Please specify a jar or directory for JUMP_CLASS_PATH'
jnius_config.set_classpath(jnius_classpath)

from jnius import autoclass

nd4j = autoclass('org.nd4j.linalg.factory.Nd4j')
transforms = autoclass('org.nd4j.linalg.ops.transforms.Transforms')
indexing = autoclass('org.nd4j.linalg.indexing.NDArrayIndex')
system = autoclass('java.lang.System')
integer = autoclass('java.lang.Integer')
native_ops_holder = autoclass('org.nd4j.nativeblas.NativeOpsHolder')
native_ops = native_ops_holder.getInstance().getDeviceNativeOps()

DoublePointer = autoclass('org.bytedeco.javacpp.DoublePointer')
FloatPointer = autoclass('org.bytedeco.javacpp.FloatPointer')
IntPointer = autoclass('org.bytedeco.javacpp.IntPointer')


def get_buffer_from_arr(np_arr):
    """

    :param np_arr:
    :return:
    """

    pointer_address = get_array_address(np_arr)
    pointer = native_ops.pointerForAddress(pointer_address)
    size = np_arr.size
    if np_arr.dtype == 'float64':
        as_double = DoublePointer(pointer)
        return nd4j.createBuffer(as_double, size)
    elif np_arr.dtype == 'float32':
        as_float = FloatPointer(pointer)
        return nd4j.createBuffer(as_float, size)
    elif np_arr.dtype == 'int64':
        as_int = IntPointer(pointer)
        return nd4j.createBuffer(as_int, size)


def get_array_address(np_arr):
    """
    :param np_arr: The numpy array to get the pointer address for
    :return:  the pointer address as a long
    """
    pointer, read_only_flag = np_arr.__array_interface__['data']
    return pointer


def from_np(np_arr):
    """

    :param np_arr:
    :return:
    """
    data_buffer = get_buffer_from_arr(np_arr)
    #   note here we divide the strides by 8 for numpy
    # the reason we do this is because numpy's strides are based on bytes rather than words
    return nd4j.create(data_buffer, np_arr.shape, map(lambda x: x / 8, np_arr.strides), 0)
