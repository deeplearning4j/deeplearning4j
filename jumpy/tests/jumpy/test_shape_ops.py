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


def test_reshape():
    jp.set_context_dtype('float64')

    shapes = [
        [(2, 3), (6, 1)],
        [(1, 2, 3), (3, 2)],
        [(3, 2, 1), (2, -1)],
        [(3, 1, 2), (-1, 3, 1)]
    ]

    for shape1, shape2 in shapes:
        x_np = np.random.random(shape1)
        y_np = np.reshape(x_np, shape2)
        x_jp = jp.array(x_np)
        y_jp = jp.reshape(x_jp, shape2)

        assert y_jp.shape == y_np.shape


if __name__ == '__main__':
    pytest.main([__file__])
