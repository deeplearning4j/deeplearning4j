#  /* ******************************************************************************
#   *
#   *
#   * This program and the accompanying materials are made available under the
#   * terms of the Apache License, Version 2.0 which is available at
#   * https://www.apache.org/licenses/LICENSE-2.0.
#   *
#   *  See the NOTICE file distributed with this work for additional
#   *  information regarding copyright ownership.
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
################################################################################

import numpy as np
from ..java_classes import ArrayList
from ..ndarray import array


def java2pyRDD(java_rdd, py_sc):
    '''
    Arguments

    `java_rdd`: JavaRDD<INDArray> instance
    `py_sc`: Pyspark context instance

    Returns

    pyspark.RDD instance
    '''
    indarray_list = java_rdd.collect()
    num_arrays = indarray_list.size()

    nparray_list = []
    for i in range(num_arrays):
        indarray = indarray_list.get(i)
        jparray = array(indarray)
        nparray = jparray.numpy()
        nparray_list.append(nparray)

    return py_sc.parallelize(nparray_list)


def py2javaRDD(py_rdd, java_sc):
    '''
    Arguments

    `py_rdd`: pyspark.RDD instance
    `java_sc`: JavaSparkContext instance

    Returns

    JavaRDD<INDArray> instance
    '''
    nparray_list = py_rdd.collect()
    indarray_list = ArrayList()

    for nparray in nparray_list:
        jparray = array(nparray)
        indarray = jparray.array
        indarray_list.add(indarray)
    return java_sc.parallelize(indarray_list)
