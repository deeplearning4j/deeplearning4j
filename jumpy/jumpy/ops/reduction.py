################################################################################
# Copyright (c) 2015-2018 Skymind, Inc.
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

from .op import op
from ..java_classes import Nd4j


template = """
@op
def {}(arr, axis=None):
    if axis is None:
        return Nd4j.{}(arr)
    return Nd4j.{}(arr)
"""

reduction_ops = [['max'], ['min'], ['sum'], ['prod'], ['mean'], [
    'std'], ['var'], ['argmax', 'argMax'], ['argmin', 'argMin']]

for rop in reduction_ops:
    code = template.format(rop[0], rop[-1], rop[-1])
    exec(code)
