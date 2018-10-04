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


def test_transpose():
    shapes = [(2, 3), (3, 1), (2, 3, 4)]
    for shape in shapes:
        x_np = np.random.random(shape)
        x_jp = jp.array(x_np)

        y_np = np.transpose(x_np)
        y_jp = jp.transpose(x_jp)

        y_jp = y_jp.numpy()

        assert y_jp.shape == y_np.shape


def test_permute():
    shapes = [
        [(2, 3), [0, 1], [1, 0]],
        [(2, 1), [0, 1], [1, 0]], 
        [(2, 3, 4), [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        ]

    for shape in shapes:
        x_np = np.random.random(shape[0])
        x_jp = jp.array(x_np)
        for dims in shape[1:]:
            y_np = np.transpose(x_np, dims)
            y_jp = jp.transpose(x_jp, dims)
            assert y_jp.shape == y_np.shape


def test_expand_dims():
    shapes = [(2, 3), (2, 1), (2, 3, 4)]
    for shape in shapes:
        x_np = np.random.random(shape)
        x_jp = jp.array(x_np)
        for axis in range(len(shape) + 1):
            y_np = np.expand_dims(x_np, axis)
            y_jp = jp.expand_dims(x_jp, axis)
            assert y_jp.shape == y_np.shape


def test_squeeze():
    shapes = [[2, 3, 1, 4], [2, 1, 3]]
    for shape in shapes:
        x_np = np.random.random(shape)
        x_jp = jp.array(x_np)
        axis = shape.index(1)
        y_np = np.squeeze(x_np, axis)
        y_jp = jp.squeeze(x_jp, axis)
        assert y_jp.shape == y_np.shape 


def test_concatenate():
    shapes = [
        [(2, 3, 4), (3, 3, 4), 0],
        [(2, 3, 5), (2, 4, 5), 1],
        [(3, 2, 4), (3, 2, 2), 2]
    ]

    for shape in shapes:
        x1_np = np.random.random(shape[0])
        x2_np = np.random.random(shape[1])

        x1_jp = jp.array(x1_np)
        x2_jp = jp.array(x2_np)

        axis = shape[2]

        y_np = np.concatenate([x1_np, x2_np], axis)
        y_jp = jp.concatenate([x1_jp, x2_jp], axis)

        assert y_jp.shape == y_np.shape 


if __name__ == '__main__':
    pytest.main([__file__])
