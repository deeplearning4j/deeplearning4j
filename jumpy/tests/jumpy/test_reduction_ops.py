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


import pytest

import jumpy as jp
import numpy as np
from numpy.testing import assert_allclose



def _test_reduction_op(op, shape):
    for axis in range(len(shape)):
        x_np = np.random.random(shape)
        y_np = getattr(np, op)(x_np, axis=axis)

        x_jp = jp.array(x_np)
        y_jp = getattr(jp, op)(x_jp, axis=axis)

        x_jp = x_jp.numpy()

        assert_allclose(x_jp, x_np)


def test_reduction_ops():
    shapes = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]   
    reduction_ops = ['max', 'min', 'sum', 'prod', 'mean', 
                     'std', 'var', 'argmax', 'argmin']

    for op in reduction_ops:
        for shape in shapes:
            _test_reduction_op(op, shape)


if __name__ == '__main__':
    pytest.main([__file__])
