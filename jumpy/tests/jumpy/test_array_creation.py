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


def test_array_creation():
    a = jp.zeros((32, 10))
    assert int(jp.sum(a)) == 0
    a = jp.ones((32, 12))
    assert int(jp.sum(a)) == 32 * 12


if __name__ == '__main__':
    pytest.main([__file__])
