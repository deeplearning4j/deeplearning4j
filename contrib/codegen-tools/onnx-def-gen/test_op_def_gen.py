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

from onnx_tf.common import attr_converter,attr_translator
from onnx_tf.handlers.backend import *
import onnx_tf
import onnx_tf.handlers.handler
import sys,inspect
import tensorflow as tf
from onnx_tf.backend import TensorflowBackend


current_module = sys.modules['onnx_tf.handlers.backend']
modules = inspect.getmembers(current_module)
for name, obj in modules:
 obj_modules = inspect.getmembers(obj)
 for name2,module2 in obj_modules:
    if inspect.isclass(module2):
        result = module2
        print(module2)