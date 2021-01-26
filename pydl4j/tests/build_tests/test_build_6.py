#  /* ******************************************************************************
#   * Copyright (c) 2021 Deeplearning4j Contributors
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
