import pytest
import pydl4j
import os


def _datavec_test():
    from jnius import autoclass
    SparkConf = autoclass('org.apache.spark.SparkConf')
    SparkContext = autoclass('org.apache.spark.api.java.JavaSparkContext')
    JavaRDD = autoclass('org.apache.spark.api.java.JavaRDD')


def test_build():
    _CONFIG = {
        'dl4j_version': '1.0.0-SNAPSHOT',
        'dl4j_core': True,
        'datavec': True,
        'spark': False,
        'spark_version': '2',
        'scala_version': '2.11',
        'nd4j_backend': 'cpu'
    }

    my_dir = pydl4j.jarmgr._MY_DIR

    if os.path.isdir(my_dir):
        os.remove(my_dir)

    pydl4j.set_config(_CONFIG)

    pydl4j.maven_build()

    import jumpy as jp

    assert jp.zeros((3, 2)).numpy().sum() == 0

    _datavec_test()


if __name__ == '__main__':
    pytest.main([__file__])
