################################################################################
# Copyright (c) 2015-2019 Skymind, Inc.
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
from ..java_classes import DatasetDescriptor as getDatasetDescriptor
from ..java_classes import DataType
from ..java_classes import spark_utils as get_spark_utils
from ..java_classes import JDataset
from ..ndarray import array
from .utils import np2desc
from .utils import py2j_ds_desc
from .utils import j2py_ds_desc
from .utils import j2py_arr_desc
from .utils import py2j_arr_desc
from .utils import desc2np
from .utils import desc2ds
from .utils import ds2desc


ArrayDescriptor = None
JDatasetDescriptor = None
spark_utils = None




def java2pyArrayRDD(java_rdd, py_sc):
    '''
    Arguments

    `java_rdd`: JavaRDD<INDArray> instance
    `py_sc`: Pyspark context instance

    Returns

    pyspark.RDD instance
    '''
    global spark_utils
    if spark_utils is None:
        spark_utils = get_spark_utils()
    desc_rdd = spark_utils.getArrayDescriptorRDD(java_rdd)
    descriptors = desc_rdd.collect()
    num_descriptors = descriptors.size()
    nparrays = []
    pydescriptors = []
    for i in range(num_descriptors):
        jdesc = descriptors.get(i)
        pydesc = j2py_arr_desc(jdesc)
        nparrays.append(desc2np(pydesc))
        #pydescriptors.append(pydesc)
    #pyrdd = py_sc.parallelize(pydescriptors)
    #pyrdd = pyrdd.map(desc2np)
    pyrdd = py_sc.parallelize(nparrays)
    return pyrdd


def py2javaArrayRDD(py_rdd, java_sc):
    '''
    Arguments

    `py_rdd`: pyspark.RDD instance
    `java_sc`: JavaSparkContext instance

    Returns

    JavaRDD<INDArray> instance
    '''
    global ArrayDescriptor, spark_utils
    if ArrayDescriptor is None:
        ArrayDescriptor = getArrayDescriptor()
    if spark_utils is None:
        spark_utils = get_spark_utils()

    #desc_rdd = py_rdd.map(np2desc)
    #descriptors = desc_rdd.collect()
    arrlist = ArrayList()
    nparrays = py_rdd.collect()
    for nparr in nparrays:
        arrlist.add(array(nparr).array)
    return java_sc.parallelize(arrlist)
    for d in descriptors:
        #arrlist.add(array(desc2np(d)).array)
        arrlist.add(ArrayDescriptor(d[0], d[1], d[2], dtype_map[d[3]], 'c').getArray())
    java_rdd = java_sc.parallelize(arrlist)
    #return java_rdd
    java_rdd = spark_utils.getArrayRDD(java_rdd)
    return java_rdd


def java2pyDatasetRDD(java_rdd, py_sc):
    global spark_utils, JDatasetDescriptor
    if spark_utils is None:
        spark_utils = get_spark_utils()
    if JDatasetDescriptor is None:
        JDatasetDescriptor = getDatasetDescriptor()
    jdatasets = java_rdd.collect()
    num_ds = jdatasets.size()
    pydatasets = []
    for i in range(num_ds):
        jds = jdatasets.get(i)
        jdesc = JDatasetDescriptor(jds)
        pydesc = j2py_ds_desc(jdesc)
        pyds = desc2ds(pydesc)
        pydatasets.append(pyds)
    return py_sc.parallelize(pydatasets)


    ####
    desc_rdd = spark_utils.getDataSetDescriptorRDD(java_rdd)
    descriptors = desc_rdd.collect()
    num_descriptors = descriptors.size()
    pydescriptors = []
    for i in range(num_descriptors):
        jdesc = descriptors.get(i)
        pydesc = j2py_ds_desc(jdesc)
        pydescriptors.append(pydesc)
    pyrdd = py_sc.parallelize(pydescriptors)
    pyrdd = pyrdd.map(desc2ds)
    return pyrdd


def py2javaDatasetRDD(py_rdd, java_sc):
    global spark_utils
    if spark_utils is None:
        spark_utils = get_spark_utils()
    
    ###
    pydatasets = py_rdd.collect()
    jdatasets = ArrayList()
    for pyds in pydatasets:
        pydesc = ds2desc(pyds)
        jdesc = py2j_ds_desc(pydesc)
        jds = jdesc.getDataSet()
        jdatasets.add(jds)
    return java_sc.parallelize(jdatasets)

    ###
    desc_rdd = py_rdd.map(ds2desc)
    pydescriptors = desc_rdd.collect()
    jdescriptors = ArrayList()
    for pydesc in pydescriptors:
        jdescriptors.add(py2j_ds_desc(pydesc))
    java_rdd = java_sc.parallelize(jdescriptors)
    java_rdd = spark_utils.getDataSetRDD(java_rdd)
    return java_rdd
