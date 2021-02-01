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
from pydatavec import Schema, TransformProcess


def test_rename():
    schema = Schema()
    schema.add_string_column('str1')

    tp = TransformProcess(schema)
    tp.rename_column('str1', 'str2')

    assert 'str1' not in tp.final_schema.columns
    assert 'str2' in tp.final_schema.columns

    tp.to_java()


def test_remove():
    schema = Schema()
    schema.add_string_column('str1')
    schema.add_string_column('str2')

    tp = TransformProcess(schema)
    tp.remove_column('str1')

    assert list(tp.final_schema.columns.keys()) == ['str2']

    tp.to_java()


def test_remove_except():
    schema = Schema()
    schema.add_string_column('str1')
    schema.add_string_column('str2')
    schema.add_string_column('str3')

    tp = TransformProcess(schema)
    tp.remove_columns_except('str2')

    assert list(tp.final_schema.columns.keys()) == ['str2']

    tp.to_java()


def test_str_to_time():
    schema = Schema()
    schema.add_string_column('str1')
    schema.add_string_column('str2')

    tp = TransformProcess(schema)

    tp.string_to_time('str1')

    assert tp.final_schema.get_column_type('str1') == 'DateTime'

    tp.to_java()


def test_derive_col_from_time():
    schema = Schema()
    schema.add_string_column('str1')
    schema.add_string_column('str2')

    tp = TransformProcess(schema)

    tp.string_to_time('str1')
    tp.derive_column_from_time('str1', 'hour', 'hour_of_day')

    assert 'hour' in tp.final_schema.columns

    tp.to_java()


def test_cat_to_int():
    schema = Schema()
    schema.add_categorical_column('cat', ['A', 'B', 'C'])

    tp = TransformProcess(schema)
    tp.categorical_to_integer('cat')

    assert tp.final_schema.get_column_type('cat') == 'integer'

    tp.to_java()


def test_append_string():
    schema = Schema()
    schema.add_string_column('str1')

    tp = TransformProcess(schema)
    tp.append_string('str1', 'xxx')

    tp.to_java()


def test_lower():
    schema = Schema()
    schema.add_string_column('str1')

    tp = TransformProcess(schema)
    tp.lower('str1')

    tp.to_java()


def test_upper():
    schema = Schema()
    schema.add_string_column('str1')

    tp = TransformProcess(schema)
    tp.upper('str1')

    tp.to_java()


def test_concat():
    schema = Schema()
    schema.add_string_column('str1')
    schema.add_string_column('str2')

    tp = TransformProcess(schema)
    tp.concat(['str1', 'str2'], 'str3')

    assert 'str3' in tp.final_schema.columns

    tp.to_java()


def test_remove_white_spaces():
    schema = Schema()
    schema.add_string_column('str1')

    tp = TransformProcess(schema)
    tp.remove_white_spaces('str1')

    tp.to_java()


def test_replace_empty():
    schema = Schema()
    schema.add_string_column('str1')

    tp = TransformProcess(schema)
    tp.replace_empty_string('str1', 'xx')

    tp.to_java()


if __name__ == '__main__':
    pytest.main([__file__])
