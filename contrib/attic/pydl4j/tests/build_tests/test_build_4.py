#  /* ******************************************************************************
#   *
#   *
#   * This program and the accompanying materials are made available under the
#   * terms of the Apache License, Version 2.0 which is available at
#   * https://www.apache.org/licenses/LICENSE-2.0.
#   *
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
# SPDX-License-Identifier: Apache-2.0
################################################################################

import pytest
import pydl4j
import os


def _spark_test():
    from jnius import autoclass
    SparkConf = autoclass('org.apache.spark.SparkConf')
    SparkContext = autoclass('org.apache.spark.api.java.JavaSparkContext')
    JavaRDD = autoclass('org.apache.spark.api.java.JavaRDD')
    SparkTransformExecutor = autoclass('org.datavec.spark.'
                                       'transform.SparkTransformExecutor')
    StringToWritablesFunction = autoclass('org.datavec.spark.'
                                          'transform.misc.'
                                          'StringToWritablesFunction')
    WritablesToStringFunction = autoclass('org.datavec.spark.'
                                          'transform.misc.'
                                          'WritablesToStringFunction')

    spark_conf = SparkConf()
    spark_conf.setMaster('local[*]')
    spark_conf.setAppName('test')

    spark_context = SparkContext(spark_conf)
    source = 'basic_example.csv'
    assert os.path.isfile(source)
    string_data = spark_context.textFile(source)


def test_build():
    _CONFIG = {
        'dl4j_version': '1.0.0-SNAPSHOT',
        'dl4j_core': True,
        'datavec': True,
        'spark': True,
        'spark_version': '1',
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

    _spark_test()


if __name__ == '__main__':
    pytest.main([__file__])
