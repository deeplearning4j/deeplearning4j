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

from numpy.testing import assert_allclose
from jumpy.spark import py2javaArrayRDD
from jumpy.spark import py2javaDatasetRDD
from jumpy.spark import java2pyArrayRDD
from jumpy.spark import java2pyDatasetRDD
from jumpy.java_classes import JDataset
from jumpy.spark import Dataset
from jumpy.java_classes import ArrayList
from numpy.testing import assert_allclose
from jnius import autoclass
import jumpy as jp
import numpy as np
import pyspark
import pytest



SparkConf = autoclass('org.apache.spark.SparkConf')
SparkContext = autoclass('org.apache.spark.api.java.JavaSparkContext')



class TestSparkConverters(object):
    
    @pytest.fixture(scope='module')
    def java_sc(self):
        config = SparkConf()
        config.setAppName("test")
        config.setMaster("local[*]")
        return SparkContext(config)

    @pytest.fixture(scope='module')
    def py_sc(self):
        return pyspark.SparkContext(master='local[*]', appName='test')

    def test_java2py_array(self, java_sc, py_sc):
        data = ArrayList()

        for _ in range(100):
            arr = jp.array(np.random.random((32, 20))).array
            data.add(arr)

        java_rdd = java_sc.parallelize(data)
        py_rdd = java2pyArrayRDD(java_rdd, py_sc)

        data2 = py_rdd.collect()

        data = [data.get(i) for i in range(data.size())]

        assert len(data) == len(data2)

        for d1, d2 in zip(data, data2):
            assert_allclose(jp.array(d1).numpy(), d2)


    def test_py2java_array(self, java_sc, py_sc):
        data = [np.random.random((32, 20)) for _ in range(100)]

        jdata = [jp.array(x) for x in data]  # required

        py_rdd = py_sc.parallelize(data)
        java_rdd = py2javaArrayRDD(py_rdd, java_sc)

        data2 = java_rdd.collect()
        data2 = [data2.get(i) for i in range(data2.size())]
        assert len(data) == len(data2)
        for d1, d2 in zip(data, data2):
            d2 = jp.array(d2).numpy()
            assert_allclose(d1, d2)

    def test_java2py_dataset(self, java_sc, py_sc):
        data = ArrayList()

        for _ in range(100):
            arr = jp.array(np.random.random((32, 20))).array
            ds = JDataset(arr, arr)
            data.add(ds)

        java_rdd = java_sc.parallelize(data)
        py_rdd = java2pyDatasetRDD(java_rdd, py_sc)

        data2 = py_rdd.collect()

        data = [data.get(i) for i in range(data.size())]

        assert len(data) == len(data2)

        for d1, d2 in zip(data, data2):
            assert_allclose(jp.array(d1.getFeatures()).numpy(), d2.features.numpy())

    def test_py2java_array(self, java_sc, py_sc):
        data = [np.random.random((32, 20)) for _ in range(100)]
        jdata = [jp.array(x) for x in data]  # required
        data = [Dataset(x, x) for x in data]
        

        py_rdd = py_sc.parallelize(data)
        java_rdd = py2javaDatasetRDD(py_rdd, java_sc)

        data2 = java_rdd.collect()
        data2 = [data2.get(i) for i in range(data2.size())]
        assert len(data) == len(data2)
        for d1, d2 in zip(data, data2):
            d2 = jp.array(d2.getFeatures()).numpy()
            assert_allclose(d1.features.numpy(), d2)



if __name__ == '__main__':
    pytest.main([__file__])
