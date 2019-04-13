import pytest
import pydl4j
import os


def test_build():
    _CONFIG = {
        'dl4j_version': '1.0.0-SNAPSHOT',
        'dl4j_core': True,
        'datavec': False,
        'spark': True,
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


if __name__ == '__main__':
    pytest.main([__file__])
