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
from ..java_classes import ArrayList
from ..java_classes import ArrayDescriptor as getArrayDescriptor
from ..java_classes import DataSetDescriptor as getDataSetDescriptor
from ..java_classes import DataType
from ..java_classes import spark_utils as get_spark_utls
from ..java_classes import DataSet
from ..ndarray import array

ArrayDescriptor = None
DataSetDescriptor = None
spark_utlis = None




def java2pyRDD(java_rdd, py_sc):
    '''
    Arguments

    `java_rdd`: JavaRDD<INDArray> instance
    `py_sc`: Pyspark context instance

    Returns

    pyspark.RDD instance
    '''
    global spark_utlis
    if spark_utlis is None:
        spark_utlis = get_spark_utls()
    desc_rdd = spark_utils.getArrayDescriptorRDD(java_rdd)
    descriptors = desc_rdd.collect()
    num_descriptors = descriptors.size()
    nparrays = []
    for i in range(num_descriptors):
        desc = descriptors.get(i)
        indarray = desc.getArray()
        jparray = array(indarray)
        nparray = jparray.numpy()
        nparrays.append(nparray)
    return py_sc.parallelize(nparrays)


def py2javaRDD(py_rdd, java_sc):
    '''
    Arguments

    `py_rdd`: pyspark.RDD instance
    `java_sc`: JavaSparkContext instance

    Returns

    JavaRDD<INDArray> instance
    '''
    global ArrayDescriptor
    if ArrayDescriptor is None:
        ArrayDescriptor = getArrayDescriptor()
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
    
    dtype_map = {
        "float": DataType.FLOAT,
        "double": DataType.DOUBLE
    }
    desc_rdd = py_rdd.map(np2desc)
    descriptors = desc_rdd.collect()
    arrlist = ArrayList()
    for d in descriptors:
        arrlist.add(ArrayDescriptor(d[0], d[1], d[2], dtype_map[d[3]]))
    java_rdd = java_sc.parallelize(arrlist)
    java_rdd = spark_utils.getArrayRDD(java_rdd)
    return java_rdd
