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
from pydatavec import Schema


def test_schema():
    schema = Schema()
    schema.add_string_column('str1')
    schema.add_string_column('str2')
    schema.add_integer_column('int1')
    schema.add_integer_column('int2')
    schema.add_double_column('dbl1')
    schema.add_double_column('dbl2')
    schema.add_float_column('flt1')
    schema.add_float_column('flt2')
    schema.add_categorical_column('cat1', ['A', 'B', 'C'])
    schema.add_categorical_column('cat2', ['A', 'B', 'C'])
    schema.to_java()


if __name__ == '__main__':
    pytest.main([__file__])
