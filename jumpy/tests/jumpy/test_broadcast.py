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


def _test_ufunc(op, shape1, shape2):
    a_np = np.random.random(shape1)
    b_np = np.random.random(shape2)

    c_np = eval('a_np {} b_np'.format(op))

    a_jp = jp.array(a_np)
    b_jp = jp.array(b_np)

    c_jp = eval('a_jp {} b_jp'.format(op))

    c_jp = c_jp.numpy()

    assert_allclose(c_jp, c_np)



def _test_ufunc_inplace(op, shape1, shape2):
    a_np = np.random.random(shape1)
    b_np = np.random.random(shape2)
    a_np2 = a_np.copy()
    exec('a_np {}= b_np'.format(op))

    a_jp = jp.array(a_np2)
    b_jp = jp.array(b_np)

    exec('a_jp {}= b_jp'.format(op))

    a_jp = a_jp.numpy()

    assert_allclose(a_jp, a_np)


def test_broadcast():
    jp.set_context_dtype('float64')
    shapes = [
        [(2, 3), (3, )],
        [(2, 3, 4), (3, 4)],
        [(2, 3), (1, 1)],
        [(2, 3), (1, 1, 1)]
    ]

    ops = ['+', '-', '*', '/']
    for op in ops:
        for shape in shapes:
            _test_ufunc(op, *shape)
            _test_ufunc(op, *reversed(shape))
            if len(shape[0]) > len(shape[1]):
                _test_ufunc_inplace(op, *shape)


if __name__ == '__main__':
    pytest.main([__file__])