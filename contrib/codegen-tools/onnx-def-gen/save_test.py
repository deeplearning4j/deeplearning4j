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

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def graph_as_func():
    S = tf.Variable(tf.constant([1, 2, 3, 4]))
    result = tf.scatter_add(S, [0], [10])
    return result
a_function_that_uses_a_graph = tf.function(graph_as_func)
print(a_function_that_uses_a_graph.__attr__)
converted = convert_variables_to_constants_v2(a_function_that_uses_a_graph)
print(type(converted))