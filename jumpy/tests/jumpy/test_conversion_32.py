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


def test_conversion_32():
    jp.set_context_dtype('float32')
    shapes = [(1, 1), (2, 1), (1, 2),  (32, 12), (100, 32, 16)]
    for shape in shapes:
        x_np = np.random.random(shape)
        x_np = np.cast['float32'](x_np)
        x_jp = jp.array(x_np)
        x_np += np.cast['float32'](np.random.random(shape))
        x_jp = x_jp.numpy()

        assert_allclose(x_jp, x_np)


if __name__ == '__main__':
    pytest.main([__file__])
