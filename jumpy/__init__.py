import jnius_config
import os


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

    Create an nd4j ndarray from a numpy array (passing the
    numpy pointer buffer by reference)

    :param np_arr: a numpy array
    :return:
    """

    import numpy as np
    # nd4j doesn't have 1d arrays. Convert to a row vector
    if np_arr.ndim == 1:
        np_arr = np.reshape(np_arr, (1, np_arr.size))

    data_buffer = get_buffer_from_arr(np_arr)
    #   note here we divide the strides by 8 for numpy
    # the reason we do this is because numpy's strides are based on bytes rather than words
    return nd4j.create(data_buffer, np_arr.shape, map(lambda x: x / 8, np_arr.strides), 0)


