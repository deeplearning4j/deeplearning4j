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


class Condition(object):

    @property
    def name(self):
        return self.__class__.__name__


class InSet(Condition):
    def __init__(self, column, set):
        self.column = column
        self.set = set


class NotInSet(Condition):
    def __init__(self, column, set):
        self.column = column
        self.set = set


class Equals(Condition):
    def __init__(self, column, value):
        self.column = column
        self.value = value


class NotEquals(Condition):
    def __init__(self, column, value):
        self.column = column
        self.value = value


class LessThan(Condition):
    def __init__(self, column, value):
        self.column = column
        self.value = value


class LessThanOrEqual(Condition):
    def __init__(self, column, value):
        self.column = column
        self.value = value


class GreaterThan(Condition):
    def __init__(self, column, value):
        self.column = column
        self.value = value


class GreaterThanOrEqual(Condition):
    def __init__(self, column, value):
        self.column = column
        self.value = value
