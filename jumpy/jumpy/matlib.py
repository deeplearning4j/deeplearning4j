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


from .ndarray import ndarray
from .java_classes import Nd4j


def zeros(shape):
    return ndarray(Nd4j.zeros(*shape))


def ones(shape):
    return ndarray(Nd4j.ones(*shape))


def zeros_like(array):
    array = ndarray(array).array
    return ndarray(Nd4j.zerosLike(array))


def ones_like(array):
    array = ndarray(array).array
    return ndarray(Nd4j.onesLike(array))


def eye(size):
    return ndarray(Nd4j.eye(size))


def arange(m, n=None):
    if n is None:
        return ndarray(Nd4j.arange(m))
    return ndarray(Nd4j.arange(m, n))


def linspace(start, stop, num):
    return ndarray(Nd4j.linspace(start, stop, num))
