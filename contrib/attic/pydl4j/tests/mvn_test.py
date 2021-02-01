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
from pydl4j import *


def test_get_artifacts():
    artifacts = get_artifacts('datavec')
    expected = ['datavec-api', 'datavec-local', 'datavec-parent']
    for e in expected:
        assert e in artifacts


def test_get_versions():
    versions = get_versions('datavec', 'datavec-api')
    assert len(versions) >= 12


def test_get_latest_version():
    v = get_latest_version('datavec', 'datavec-api')
    assert len(v) > 0


def test_install():
    set_context('test')
    clear_context()
    mvn_install('datavec', 'datavec-api')
    mvn_install('datavec', 'datavec-local')
    assert len(get_jars()) == 2
    for jar in get_jars():
        uninstall(jar)
    assert len(get_jars()) == 0
    clear_context()


if __name__ == '__main__':
    pytest.main([__file__])
