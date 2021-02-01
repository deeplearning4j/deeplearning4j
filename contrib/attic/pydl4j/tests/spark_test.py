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
import jnius_config
import os
import warnings
import pydl4j


def test_spark():
    # skip test in travis
    if "TRAVIS" in os.environ and os.environ["TRAVIS"] == "true":
        return

    pydl4j.validate_datavec_jars()

    from jnius import autoclass

    SparkConf = autoclass('org.apache.spark.SparkConf')
    SparkContext = autoclass('org.apache.spark.api.java.JavaSparkContext')
    JavaRDD = autoclass('org.apache.spark.api.java.JavaRDD')
    SparkTransformExecutor = autoclass(
        'org.datavec.spark.transform.SparkTransformExecutor')
    StringToWritablesFunction = autoclass(
        'org.datavec.spark.transform.misc.StringToWritablesFunction')
    WritablesToStringFunction = autoclass(
        'org.datavec.spark.transform.misc.WritablesToStringFunction')

    spark_conf = SparkConf()
    spark_conf.setMaster('local[*]')
    spark_conf.setAppName('test')

    spark_context = SparkContext(spark_conf)
    source = 'basic_example.csv'
    assert os.path.isfile(source)
    string_data = spark_context.textFile(source)


if __name__ == '__main__':
    pytest.main([__file__])
