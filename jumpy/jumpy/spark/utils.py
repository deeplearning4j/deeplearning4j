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


import numpy as np
from ..java_classes import DataType
from ..java_classes import ArrayDescriptor as getArrayDescriptor
from ..java_classes import DatasetDescriptor
import ctypes
from .dataset import Dataset


ArrayDescriptor = None


def np2desc(nparray):
    address = nparray.__array_interface__['data'][0]
    shape = nparray.shape
    stride = nparray.strides
    nptype = nparray.dtype
    if nptype == np.float32:
        dtype = "float"
    elif nptype == np.float64:
        dtype = "double"
    else:
        raise Exception("Unsupported data type: " + str(nptype))
    return (address, shape, stride, dtype)


def desc2np(desc):
    address, shape, stride, dtype = desc
    mapping = {
        'double': ctypes.c_double,
        'float': ctypes.c_float,
        'half': ctypes.c_short,
        'long': ctypes.c_long,
        'int': ctypes.c_int,
        'short': ctypes.c_short,
        'bool': ctypes.c_bool
    }
    Pointer = ctypes.POINTER(mapping[dtype])
    pointer = ctypes.cast(address, Pointer)
    np_array = np.ctypeslib.as_array(pointer, shape)
    return np_array


def desc2ds(desc):
    return Dataset(*list(map(desc2np, desc)))


def ds2desc(ds):
    items = [ds.features, ds.labels, ds.features_mask, ds.labels_mask]
    return tuple(map(np2desc, items))


def j2py_arr_desc(jdesc):
    if jdesc is None:
        return None
    address = jdesc.getAddress()
    shape = tuple(jdesc.getShape())
    stride = tuple(jdesc.getStride())
    dtype = jdesc.getType().toString().lower()
    supported_dtypes = ["float", "double"]
    if dtype not in supported_dtypes:
        raise Exception("Unsupported data type: " + dtype)
    return (address, shape, stride, dtype)


def py2j_arr_desc(pydesc):
    global ArrayDescriptor
    if pydesc is None:
        return None
    address = pydesc[0]
    shape = pydesc[1]
    stride = pydesc[2]
    dtype = pydesc[3]
    dtype = {"float": DataType.FLOAT, "double": DataType.DOUBLE}[dtype]
    if ArrayDescriptor is None:
        ArrayDescriptor = getArrayDescriptor()
    return ArrayDescriptor(address, shape, stride, dtype)


def j2py_ds_desc(jdesc):
    jfeaturesdesc = jdesc.getFeatures()
    pyfeaturesdesc = j2py_arr_desc(jfeaturesdesc)
    jlabelsdesc = jdesc.getLabels()
    pylabelsdesc = j2py_arr_desc(jlabelsdesc)

    jfmaskdesc = jdesc.getFeaturesMask()
    pyfmaskdesc = j2py_arr_desc(jfmaskdesc)

    jlmaskdesc = jdesc.getLabelsMask()
    pylmaskdesc = j2py_arr_desc(jlmaskdesc)

    return (pyfeaturesdesc, pylabelsdesc, pyfmaskdesc, pylmaskdesc)


def py2j_ds_desc(pydesc):
    return DatasetDescriptor()(*list(map(py2j_arr_desc, pydesc)))
