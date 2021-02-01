#  /* ******************************************************************************
#   *
#   *
#   * This program and the accompanying materials are made available under the
#   * terms of the Apache License, Version 2.0 which is available at
#   * https://www.apache.org/licenses/LICENSE-2.0.
#   *
#   *  See the NOTICE file distributed with this work for additional
#   *  information regarding copyright ownership.
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#   * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#   * License for the specific language governing permissions and limitations
#   * under the License.
#   *
#   * SPDX-License-Identifier: Apache-2.0
#   ******************************************************************************/

import tensorflow as tf

dtypes = [tf.float16,tf.float32,tf.float64,tf.int8,tf.int16,tf.int32,tf.int64,tf.uint8,tf.uint16,tf.uint32,tf.uint64]
# Quick solution from https://stackoverflow.com/questions/5360220/how-to-split-a-list-into-pairs-in-all-possible-ways :)
import itertools
def all_pairs(lst):
    return [(x,y) for x in dtypes for y in dtypes]


for item in all_pairs(dtypes):
        from_dtype, out_dtype = item
        tf.reset_default_graph()
        input = tf.placeholder(name='input',dtype=from_dtype)
        result = tf.cast(input,name='cast_output',dtype=out_dtype)

        with tf.Session() as session:
            tf.train.write_graph(tf.get_default_graph(),logdir='.',name='cast_' + from_dtype.name + '_'  + out_dtype.name + '.pb',as_text=True)