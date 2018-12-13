from numpy.testing import assert_allclose
from jumpy.spark import py2javaArrayRDD
from jumpy.spark import py2javaDatasetRDD
from jumpy.spark import java2pyArrayRDD
from jumpy.spark import java2pyDatasetRDD
from jumpy.spark import Dataset
from jumpy.java_classes import ArrayList
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
            arr = jp.zeros((20, 10)).array
            data.add(arr)

        java_rdd = java_sc.parallelize(data)
        py_rdd = java2pyArrayRDD(java_rdd, py_sc)

        data2 = py_rdd.collect()
        assert len(data2) == data.size()
        assert np.sum(data2) == 0.


    def test_py2java_array(self, java_sc, py_sc):
        data = [np.zeros((20, 10)) for _ in range(100)]

        py_rdd = py_sc.parallelize(data)
        java_rdd = py2javaArrayRDD(py_rdd, java_sc)

        data2 = java_rdd.collect()
        n = data2.size()
        assert n == len(data)
        s = 0.
        for i in range(n):
            s += float(jp.sum(data2.get(i)))
        assert s == 0.



if __name__ == '__main__':
    pytest.main([__file__])
