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

################################################################################
#
#
#
################################################################################

from .java_classes import KerasModelImport
from .ndarray import array


class KerasModel(object):
    def __init__(self, filepath):
        KerasModelImport = KerasModelImport()
        try:
            self.model = KerasModelImport.importKerasModelAndWeights(filepath)
            self.is_sequential = False
        except Exception:
            self.model = KerasModelImport.importKerasSequentialModelAndWeights(filepath)
            self.is_sequential = True

    def __call__(self, input):
        if self.is_sequential:
            if type(input) in [list, tuple]:
                n = len(input)
                if n != 1:
                    err = 'Expected 1 input to sequential model. Received {}.'.format(n)
                    raise ValueError(err)
                input = input[0]
            input = array(input).array
            out = self.model.output(input, False)
            out = array(out)
            return out
        else:
            if not isinstance(input, list):
                input = [input]
            input = [array(x).array for x in input]
            out = self.model.output(False, *input)
            out = [array(x) for x in out]
            if len(out) == 1:
                return out[0]
            return out
