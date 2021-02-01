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

from .java_classes import TFGraphMapper, Nd4j, NDArrayIndex
from .ndarray import array


class TFModel(object):
    def __init__(self, filepath, inputs=None, outputs=None):
        self.sd = TFGraphMapper.getInstance().importGraph(filepath)
        self.inputs = inputs
        self.outputs = outputs
        if inputs is None:
            input_vars = [self.sd.variables().get(0)]
        elif type(inputs) in [list, tuple]:
            input_vars = []
            for x in inputs:
                var = self.sd.getVariable(x)
                if var is None:
                    raise ValueError('Variable not found in samediff graph: ' + x)
                input_vars.append(var)
        else:
            input_vars = [self.sd.getVariable(inputs)]
            if input_vars[0] is None:
                raise ValueError('Variable not found in samediff graph: ' + inputs)
        if outputs is None:
            nvars = self.sd.variables().size()
            output_vars = [self.sd.variables().get(nvars - 1)]
        elif type(outputs) in [list, tuple]:
            output_vars = []
            for x in outputs:
                var = self.sd.getVariable(x)
                if var is None:
                    raise ValueError('Variable not found in samediff graph: ' + x)
                output_vars.append(var)
        else:
            output_vars = [self.sd.getVariable(outputs)]
            if output_vars[0] is None:
                raise ValueError('Variable not found in samediff graph: ' + outputs)
        self.input_vars = input_vars
        self.output_vars = output_vars

    def __call__(self, input):
        if type(input) in (list, tuple):
            input_arrays = [array(x).array for x in input]
        else:
            input_arrays = [array(input).array]
        for arr, var in zip(input_arrays, self.input_vars):
            self.sd.associateArrayWithVariable(arr, var)
        output_arrays = []
        getattr(self.sd, 'exec')()
        for var in self.output_vars:
            output_arrays.append(array(var.getArr()))
        if len(output_arrays) == 1:
            return output_arrays[0]
        return output_arrays
