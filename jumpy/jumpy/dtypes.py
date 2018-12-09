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

from .java_classes import DataType
import numpy as np



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


POINTERS = [

]

_PY2J = {SUPPORTED_PYTHON_DTYPES[i] : SUPPORTED_JAVA_DTYPES[i] for i in range(len(SUPPORTED_JAVA_DTYPES))}
_J2PY = {SUPPORTED_JAVA_DTYPES[i] : SUPPORTED_PYTHON_DTYPES[i] for i in range(len(SUPPORTED_JAVA_DTYPES))}


def dtype_py2j(dtype):
    if isinstance(dtype, str):
        dtype = np.dtype(dtype).type
    elif isinstance(dtype, np.dtype):
        dtype = dtype.type
    jtype = _PY2J.get(dtype)
    if jtype is None:
        raise NotImplementedError("Unsupported type: " + dtype.name)
    return jtype


def dtype_j2py(dtype):
    pytype = _J2PY.get(dtype)
    if pytype is None:
        raise NotImplementedError("Unsupported type: " + (str(dtype)))
    return pytype
